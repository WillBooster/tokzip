import { CRC_INITIAL_STATE, crc32Append, crc32Finalize } from './checksum.ts';
import { pushByteVarint, pushCrc32Binary, readCrc32Binary } from './container.ts';
import { languageByName, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import { decodeFastBodyBinary, emitFastBody, fastBodyCost, fastPricing, packFastCodes } from './fastMode.ts';
import { DEFAULT_MAX_OUTPUT_SIZE, FAST_WINDOW, MODE_FAST, MODE_SMALL, MODE_STORED, SMALL_WINDOW } from './format.ts';
import { dictIndexFor, OPTIMAL_MAX_INPUT, parse } from './lz.ts';
import { TextSink } from './radix64.ts';
import { decodeSmallBodyBinary, emitSmallBody, planSmallBody, smallPricing } from './smallMode.ts';

/**
 * First byte of every tokzip stream: bit 7 set (binary channel) over low-6 magic 0b111 and
 * stream-format version 1 — disjoint from every frame magic (low-6 magic 0b110) so streams
 * and one-shot frames can never be confused.
 */
const STREAM_MAGIC_VERSION = 0b1011_1001;

/**
 * Stream flags byte: bits 1:0 carry the stream mode (fast/small); bit 2 marks window
 * carry-over (blocks are decoded with the previous blocks' output seeded as history — the
 * small-mode literal context chains across the block boundary, so decoders must know);
 * the rest are reserved.
 */
const STREAM_FLAG_CARRY = 0b100;
const STREAM_RESERVED_FLAG_MASK = 0b1111_1000;

const BYTE_VARINT_MAX_BYTES = 5;
/** The terminator's total-size varint spans the whole stream: 8 groups = 56 bits ≥ 2^53. */
const TERMINATOR_VARINT_MAX_BYTES = 8;

const DEFAULT_BLOCK_SIZE = 1 << 18; // 256 KB (matches the fast-mode window).
const MIN_BLOCK_SIZE = 1 << 10;

const textEncoder = new TextEncoder();

/** Brand check accepting genuine Uint8Arrays from any realm (iframe, vm), unlike instanceof. */
function isUint8Array(value: unknown): value is Uint8Array {
  return value instanceof Uint8Array || Object.prototype.toString.call(value) === '[object Uint8Array]';
}

export interface CompressionStreamOptions {
  /** Language dictionary to use; default 'none' (id 0, wrapper dictionary only). */
  language?: string;
  /** Optimization target; both modes are lossless. Default 'fast'. */
  mode?: 'fast' | 'small';
  /**
   * Raw bytes per compressed block. Default 256 KB, minimum 1 KB. In `fast` mode, larger
   * blocks trade latency/memory for ratio; in `small` mode the default 256 KB is the
   * practical ceiling — larger blocks shrink the carried-history budget (it keeps
   * history + block inside the optimal parser's 512 KB input bound), and at ≥ 512 KB carry
   * is impossible and the block itself exceeds the bound, degrading the parse to greedy.
   */
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
   * to the 256 KB window in `fast` mode; in `small` mode it defaults to the remaining
   * optimal-parse budget, `max(0, 512 KB − blockSize)` capped at the 1 MB window (zero at
   * `blockSize` ≥ 512 KB, where carry is impossible).
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
  private pending: Uint8Array[] = [];
  private pendingLength = 0;
  private headerWritten = false;
  /** Chained CRC over all raw bytes emitted so far (see checksum.ts). */
  private crcState = CRC_INITIAL_STATE;
  private totalRawBytes = 0;
  /** Trailing high surrogate held back from the previous string chunk (see push). */
  private pendingHighSurrogate = '';

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
    // A zero history budget (huge small-mode blocks, or an explicit historyLimit of 0)
    // degenerates to carry-less blocks; the header flag must say so, or decoders would seed
    // history the encoder never used. An explicitly requested carry must not be silently
    // dropped — fail loudly so the caller fixes the conflicting options.
    if (options?.carryWindow === true && this.historyLimit === 0) {
      throw new RangeError('carryWindow: true requires a non-zero history budget; reduce blockSize or historyLimit');
    }
    this.carryWindow = (options?.carryWindow ?? true) && this.historyLimit > 0;
  }

  push(chunk: Uint8Array | string, controller: ByteController): void {
    let bytes: Uint8Array;
    if (typeof chunk === 'string') {
      // Mirror TextEncoderStream: a surrogate pair split across chunk boundaries must not
      // become two U+FFFD, so a trailing high surrogate is held back and prepended to the
      // next string chunk (finish() flushes a leftover lone surrogate as U+FFFD).
      let text = this.pendingHighSurrogate + chunk;
      this.pendingHighSurrogate = '';
      const last = text.codePointAt(text.length - 1);
      if (last !== undefined && last >= 0xD8_00 && last <= 0xDB_FF) {
        this.pendingHighSurrogate = text.at(-1)!;
        text = text.slice(0, -1);
      }
      bytes = textEncoder.encode(text);
    } else {
      if (!isUint8Array(chunk)) throw new TypeError('chunk must be a Uint8Array or string');
      // A byte chunk ends any chance of the held-back surrogate pairing, and its bytes must
      // land BEFORE the chunk's — flush it as U+FFFD now (per WHATWG encoding, an unmatched
      // leading surrogate that cannot pair is replaced) instead of reordering it later.
      if (this.pendingHighSurrogate) {
        const flushed = textEncoder.encode(this.pendingHighSurrogate);
        this.pendingHighSurrogate = '';
        this.pending.push(flushed);
        this.pendingLength += flushed.length;
      }
      // The chunk is retained without copying until its block is emitted. That matches the
      // platform CompressionStream contract: mutating a chunk after write() is unsupported
      // and corrupts output identically there, so no defensive copy is paid here.
      bytes = chunk;
    }
    if (bytes.length > 0) {
      this.pending.push(bytes);
      this.pendingLength += bytes.length;
    }
    // Drained even when this chunk itself was empty: a flushed held-back surrogate above
    // can push the pending queue past the block size on its own.
    while (this.pendingLength >= this.blockSize) this.emitBlock(this.takeBlock(this.blockSize), controller);
  }

  finish(controller: ByteController): void {
    if (this.pendingHighSurrogate) {
      // A stream ending in a lone high surrogate encodes it as U+FFFD (TextEncoderStream's
      // flush behavior).
      const bytes = textEncoder.encode(this.pendingHighSurrogate);
      this.pendingHighSurrogate = '';
      this.pending.push(bytes);
      this.pendingLength += bytes.length;
      while (this.pendingLength >= this.blockSize) this.emitBlock(this.takeBlock(this.blockSize), controller);
    }
    if (this.pendingLength > 0) this.emitBlock(this.takeBlock(this.pendingLength), controller);
    const out = new TextSink(16);
    if (!this.headerWritten) this.writeHeader(out);
    out.push(0); // End-of-stream marker (a zero block-length varint).
    // Authenticated terminator: total raw size + final chained CRC, so dropping trailing
    // blocks (every block prefix is internally consistent) is still detected.
    pushByteVarint(out, this.totalRawBytes);
    pushCrc32Binary(out, crc32Finalize(this.crcState));
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
    // Consumed chunks are dropped in one batch after the loop: per-chunk Array.shift() would
    // make block assembly quadratic in the chunk count (e.g. byte-sized writes).
    let head = 0;
    while (at < size) {
      const chunk = this.pending[head]!;
      const take = Math.min(chunk.length, size - at);
      block.set(chunk.subarray(0, take), at);
      at += take;
      if (take === chunk.length) head++;
      else this.pending[head] = chunk.subarray(take);
    }
    if (head > 0) this.pending = this.pending.slice(head);
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
      // The small parse must finish before fastPricing runs: each pricing hands the parser a
      // module-level scratch prefix that is only valid until the next pricing call.
      const pricing = smallPricing(input, language);
      const smallTokens = parse(input, language.dictionary, dictIndex, pricing, undefined, historyLength);
      const plan = planSmallBody(smallTokens, input, language);
      const smallBytes = Math.ceil(plan.totalBits / 8);
      // Two fast candidates, mirroring the one-shot auto-downgrade: the small (optimal)
      // parse re-priced in fast units (undefined when a token exceeds fast's offset range),
      // and a pure fast parse — the optimal parse minimizes bits, so alone it could ship a
      // fast body larger than mode 'fast' would produce. Pure fast wins ties.
      const lazyFastChars = fastBodyCost(smallTokens, input, language);
      const fastPricingModel = fastPricing(input, language);
      fastPricingModel.lazy = true;
      const pureFastTokens = parse(input, language.dictionary, dictIndex, fastPricingModel, undefined, historyLength);
      const pureFastChars = fastBodyCost(pureFastTokens, input, language)!;
      const useLazyTokens = lazyFastChars !== undefined && lazyFastChars < pureFastChars;
      const fastChars = useLazyTokens ? lazyFastChars : pureFastChars;
      const fastBytes = Math.ceil((fastChars * 6) / 8);
      const best = Math.min(block.length, fastBytes, smallBytes);
      if (fastBytes === best && fastBytes < block.length) {
        mode = MODE_FAST;
        const sink = new TextSink(fastChars);
        emitFastBody(sink, useLazyTokens ? smallTokens : pureFastTokens, input, language);
        body = packFastCodes(sink.buffer, sink.length);
      } else if (smallBytes === best && smallBytes < block.length) {
        mode = MODE_SMALL;
        body = emitSmallBody(plan, language).toBytes();
      }
    }
    if (mode === MODE_STORED) body = block;

    const out = new TextSink(body!.length + 20);
    if (!this.headerWritten) this.writeHeader(out);
    pushByteVarint(out, body!.length);
    out.push(mode);
    pushByteVarint(out, block.length);
    // Chained, not per-block: the stored value is the cumulative CRC of every raw byte up
    // to and including this block, so block deletion/reordering/replay breaks the chain.
    this.crcState = crc32Append(this.crcState, block);
    this.totalRawBytes += block.length;
    pushCrc32Binary(out, crc32Finalize(this.crcState));
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
  /** Valid bytes in `buffer` are [offset, length); capacity beyond `length` is reusable. */
  private length = 0;
  private offset = 0;
  private headerSeen = false;
  private languageId = 0;
  /** Resolved lazily on the first non-stored block, so stored-only streams decode with any language id. */
  private language: RegisteredLanguage | undefined;
  private streamMode = 0;
  private carry = false;
  private window = 0;
  private history: Uint8Array = new Uint8Array(0);
  /** Chained CRC over all raw bytes produced so far (mirrors the encoder). */
  private crcState = CRC_INITIAL_STATE;
  private totalRawBytes = 0;
  private done = false;

  constructor(options?: DecompressionStreamOptions) {
    const maxBlockSize = options?.maxBlockSize ?? DEFAULT_MAX_OUTPUT_SIZE;
    // typeof, not just NaN/negative: an untyped caller passing e.g. '10MB' would compare
    // false against every size and silently disable the allocation guard.
    if (typeof maxBlockSize !== 'number' || Number.isNaN(maxBlockSize) || maxBlockSize < 0) {
      throw new RangeError(`invalid maxBlockSize: ${maxBlockSize}`);
    }
    this.maxBlockSize = maxBlockSize;
  }

  push(chunk: Uint8Array, controller: ByteController): void {
    if (!isUint8Array(chunk)) throw new TypeError('chunk must be a Uint8Array');
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
    return this.length - this.offset;
  }

  /**
   * Appends into a growable buffer with amortized-doubling capacity, compacting consumed
   * bytes in place first — reallocating per chunk would make block assembly quadratic in
   * the chunk count.
   */
  private append(chunk: Uint8Array): void {
    const remaining = this.length - this.offset;
    if (this.buffer.length - this.length < chunk.length) {
      if (this.buffer.length - remaining >= chunk.length && this.offset > 0) {
        this.buffer.copyWithin(0, this.offset, this.length);
      } else {
        const grown = new Uint8Array(Math.max(remaining + chunk.length, this.buffer.length * 2, 16_384));
        grown.set(this.buffer.subarray(this.offset, this.length));
        this.buffer = grown;
      }
      this.length = remaining;
      this.offset = 0;
    }
    this.buffer.set(chunk, this.length);
    this.length += chunk.length;
  }

  /** Reads a canonical byte varint at `pos`, or undefined when more input is needed. */
  private tryReadVarint(pos: number, maxBytes = BYTE_VARINT_MAX_BYTES): { value: number; pos: number } | undefined {
    let value = 0;
    let shift = 1;
    for (let i = 0; i < maxBytes; i++) {
      if (pos >= this.length) return undefined;
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
    if (!this.headerSeen) {
      if (this.available() < 3) return false;
      const magic = this.buffer[this.offset]!;
      if (magic !== STREAM_MAGIC_VERSION) {
        if ((magic & 0b1111_1000) === (STREAM_MAGIC_VERSION & 0b1111_1000)) {
          throw new TokzipDecodeError('unknown version');
        }
        throw new TokzipDecodeError('bad magic');
      }
      this.languageId = this.buffer[this.offset + 1]!;
      const flags = this.buffer[this.offset + 2]!;
      if ((flags & STREAM_RESERVED_FLAG_MASK) !== 0) throw new TokzipDecodeError('reserved flag bits set');
      const mode = flags & 3;
      if (mode !== MODE_FAST && mode !== MODE_SMALL) throw new TokzipDecodeError('invalid mode');
      this.headerSeen = true;
      this.streamMode = mode;
      this.carry = (flags & STREAM_FLAG_CARRY) !== 0;
      this.window = mode === MODE_FAST ? FAST_WINDOW : SMALL_WINDOW;
      this.offset += 3;
      return true;
    }

    const lengthField = this.tryReadVarint(this.offset);
    if (!lengthField) return false;
    const bodyLength = lengthField.value;
    if (bodyLength === 0) {
      // Authenticated terminator: total raw size + final chained CRC. Verifying it catches
      // trailing-block deletion, which every per-block check necessarily misses. The total
      // spans the whole stream, beyond the per-block 35-bit bound — 8 groups cover 2^53
      // (Number.MAX_SAFE_INTEGER), which the byte counter cannot exceed anyway.
      const totalField = this.tryReadVarint(lengthField.pos, TERMINATOR_VARINT_MAX_BYTES);
      if (!totalField) return false;
      if (this.length - totalField.pos < 4) return false;
      const declaredCrc = readCrc32Binary(this.buffer, totalField.pos);
      if (totalField.value !== this.totalRawBytes) throw new TokzipDecodeError('stream length mismatch');
      if (declaredCrc !== crc32Finalize(this.crcState)) throw new TokzipDecodeError('checksum mismatch');
      this.offset = totalField.pos + 4;
      this.done = true;
      if (this.available() > 0) throw new TokzipDecodeError('trailing bytes after end of stream');
      return false;
    }
    if (lengthField.pos >= this.length) return false;
    const mode = this.buffer[lengthField.pos]!;
    const rawField = this.tryReadVarint(lengthField.pos + 1);
    if (!rawField) return false;
    const rawLength = rawField.value;
    // The 4-byte block CRC sits between the raw-length varint and the body.
    const bodyStart = rawField.pos + 4;

    // Every prefix-derived constraint is enforced BEFORE waiting for the body: a hostile
    // stream declaring a huge body must fail here, not after the decoder has buffered it —
    // these checks are what bound buffering by maxBlockSize.
    if (rawLength === 0) throw new TokzipDecodeError('empty block');
    if (rawLength > this.maxBlockSize) throw new TokzipDecodeError('declared size exceeds maxBlockSize');
    if (mode === MODE_STORED) {
      if (bodyLength !== rawLength) throw new TokzipDecodeError('stored block length mismatch');
    } else if (mode === MODE_FAST || mode === MODE_SMALL) {
      // A fast-mode stream retains only the fast window and its encoder never emits small
      // blocks, so a small block under a fast header is non-canonical (and could reference
      // history beyond the retained window).
      if (mode === MODE_SMALL && this.streamMode === MODE_FAST) {
        throw new TokzipDecodeError('small block in fast stream');
      }
      // Mirrors the frame containers: a conforming non-stored body is strictly smaller than
      // the stored body, keeping blocks canonical and allocations bounded.
      if (bodyLength >= rawLength) throw new TokzipDecodeError('non-canonical block: body not smaller than stored');
    } else {
      throw new TokzipDecodeError('invalid mode');
    }
    if (this.length - bodyStart < bodyLength) return false;
    const declaredCrc = readCrc32Binary(this.buffer, bodyStart - 4);

    const bodyEnd = bodyStart + bodyLength;
    let out: Uint8Array;
    if (mode === MODE_STORED) {
      out = this.buffer.slice(bodyStart, bodyEnd);
    } else {
      const language = (this.language ??= requireLanguageById(this.languageId));
      const history = this.carry && this.history.length > 0 ? this.history : undefined;
      out =
        mode === MODE_FAST
          ? decodeFastBodyBinary(this.buffer, bodyStart, bodyEnd, rawLength, language, false, history)
          : decodeSmallBodyBinary(this.buffer, bodyStart, bodyEnd, rawLength, language, false, history);
    }
    this.crcState = crc32Append(this.crcState, out);
    this.totalRawBytes += out.length;
    if (crc32Finalize(this.crcState) !== declaredCrc) throw new TokzipDecodeError('checksum mismatch');
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
