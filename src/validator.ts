import { readByteVarint, readCrc32Binary, readCrc32Text } from './container.ts';
import { TokzipDecodeError } from './errors.ts';
import {
  BINARY_MAGIC_VERSION,
  CRC_BINARY_BYTES,
  CRC_TEXT_CHARS,
  FLAG_BYTES,
  FLAG_FENCED,
  MAGIC_VERSION,
  MATCH_LEN_CAP,
  MODE_FAST,
  MODE_SMALL,
  MODE_STORED,
  RESERVED_FLAG_MASK,
} from './format.ts';
import { packedRawLength, readRadix64, readVarint64 } from './radix64.ts';

/** Header facts of a structurally plausible frame (see {@link inspectFrame}). */
export interface FrameInfo {
  container: 'text' | 'binary';
  /** Wire-format version (currently always 1). */
  version: number;
  languageId: number;
  mode: 'stored' | 'fast' | 'small';
  fenced: boolean;
  contentType: 'string' | 'bytes';
  /** Declared decompressed size in bytes. */
  contentBytes: number;
  /** Declared CRC-32 of the decompressed content. */
  checksum: number;
  /** Total frame length in output units (chars for text, bytes for binary). */
  frameLength: number;
}

const MODE_NAMES = ['stored', 'fast', 'small'] as const;

/**
 * Validates a frame's header and structural envelope WITHOUT decompressing it — the
 * server-side pass-through check for deployments where only clients compress/decompress.
 * Verified: magic/version, reserved flag bits, mode, canonical size varint and checksum
 * field, and the body-length envelope (stored bodies exact, non-stored bodies strictly
 * smaller than the stored bound). NOT verified (impossible without decoding): the token
 * structure and the content checksum itself — those are enforced by `decompress` on the
 * reading client. Throws {@link TokzipDecodeError} on any violation.
 */
export function inspectFrame(data: string | Uint8Array): FrameInfo {
  return typeof data === 'string' ? inspectText(data) : inspectBinary(data);
}

function inspectText(data: string): FrameInfo {
  const magicVersion = readRadix64(data, 0);
  if (magicVersion !== MAGIC_VERSION) {
    if (magicVersion >>> 3 === MAGIC_VERSION >>> 3) throw new TokzipDecodeError('unknown version');
    throw new TokzipDecodeError('bad magic');
  }
  const languageId = readRadix64(data, 1);
  const flags = readRadix64(data, 2);
  if ((flags & RESERVED_FLAG_MASK) !== 0) throw new TokzipDecodeError('reserved flag bits set');
  const { value: contentBytes, pos: crcStart } = readVarint64(data, 3);
  const checksum = readCrc32Text(data, crcStart);
  const bodyStart = crcStart + CRC_TEXT_CHARS;
  const mode = flags & 3;
  const bodyLength = data.length - bodyStart;
  if (mode === MODE_STORED) {
    if (bodyLength < packedRawLength(contentBytes)) throw new TokzipDecodeError('truncated payload');
    if (bodyLength > packedRawLength(contentBytes)) throw new TokzipDecodeError('trailing characters after payload');
  } else if (mode === MODE_FAST || mode === MODE_SMALL) {
    // A compressed body producing content is at least one unit long; header-only frames
    // with a nonzero declared size are missing their payload.
    if (bodyLength === 0 && contentBytes > 0) throw new TokzipDecodeError('truncated payload');
    // Theoretical capacity bound, mirroring the decoders: every fast token consumes ≥ 1
    // char (small: ≥ 1 bit, 5 chars = 32 bits) and produces ≤ MATCH_LEN_CAP bytes, so a
    // larger declared size is structurally unproducible from this body.
    const capacity = mode === MODE_FAST ? bodyLength * MATCH_LEN_CAP : Math.ceil(bodyLength / 5) * 32 * MATCH_LEN_CAP;
    if (contentBytes > capacity) throw new TokzipDecodeError('declared size exceeds body capacity');
    if (bodyLength >= packedRawLength(contentBytes)) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  return frameInfo('text', languageId, flags, contentBytes, checksum, data.length);
}

function inspectBinary(data: Uint8Array): FrameInfo {
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
  if ((flags & (0b1111_0000 | RESERVED_FLAG_MASK)) !== 0) throw new TokzipDecodeError('reserved flag bits set');
  const { value: contentBytes, pos: crcStart } = readByteVarint(data, 3);
  const checksum = readCrc32Binary(data, crcStart);
  const bodyStart = crcStart + CRC_BINARY_BYTES;
  const mode = flags & 3;
  const bodyLength = data.length - bodyStart;
  if (mode === MODE_STORED) {
    if (bodyLength < contentBytes) throw new TokzipDecodeError('truncated payload');
    if (bodyLength > contentBytes) throw new TokzipDecodeError('trailing characters after payload');
  } else if (mode === MODE_FAST || mode === MODE_SMALL) {
    // Mirrors the text inspector: header-only frames with declared content are truncated.
    if (bodyLength === 0 && contentBytes > 0) throw new TokzipDecodeError('truncated payload');
    // Binary bodies pack fast chars at 6 bits and small bits at 8 per byte.
    const capacity =
      mode === MODE_FAST ? Math.floor((bodyLength * 8) / 6) * MATCH_LEN_CAP : bodyLength * 8 * MATCH_LEN_CAP;
    if (contentBytes > capacity) throw new TokzipDecodeError('declared size exceeds body capacity');
    if (bodyLength >= contentBytes) {
      throw new TokzipDecodeError('non-canonical frame: body not smaller than stored');
    }
  } else {
    throw new TokzipDecodeError('invalid mode');
  }
  return frameInfo('binary', languageId, flags, contentBytes, checksum, data.length);
}

function frameInfo(
  container: 'text' | 'binary',
  languageId: number,
  flags: number,
  contentBytes: number,
  checksum: number,
  frameLength: number
): FrameInfo {
  return {
    container,
    version: MAGIC_VERSION & 7,
    languageId,
    mode: MODE_NAMES[flags & 3]!,
    fenced: (flags & FLAG_FENCED) !== 0,
    contentType: (flags & FLAG_BYTES) !== 0 ? 'bytes' : 'string',
    contentBytes,
    checksum,
    frameLength,
  };
}
