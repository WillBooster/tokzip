import { languageById, requireLanguageById, type RegisteredLanguage } from './dictionary.ts';
import { TokzipDecodeError } from './errors.ts';
import { LANGUAGE_IDS } from './languageIds.ts';
import { dictIndexFor, type DictSegment, type Token } from './lz.ts';

/**
 * Fenced-code-block dictionary extension (FORMAT.md §6.1, flag bit 3).
 *
 * Inside a triple-backtick code fence labeled with a known language, the dictionary space is
 * extended: offsets below the frame language's assembled dictionary length keep their plain
 * meaning, and the block language's dictionary suffix is addressed contiguously above it —
 * a strict superset of the unfenced space, so fenced matching never loses the frame
 * dictionary's coverage (e.g. native-language comments inside code blocks). The fence state
 * is derived from the decoded output itself, so frames carry only the flag bit; this module
 * is the single implementation of the normative fence grammar, shared by the encoder
 * pre-scan and the decoder tracker so the two can never diverge.
 */

const BACKTICK = 0x60;
const LF = 0x0A;

/**
 * Normative fence-label aliases per language name (FORMAT.md §6.1). Labels are matched
 * ASCII-lowercased; unknown or empty labels keep the surrounding frame language.
 */
const FENCE_LABEL_ALIASES: Readonly<Record<string, readonly string[]>> = {
  c: ['c', 'h'],
  cpp: ['cpp', 'c++', 'cc', 'cxx', 'hpp'],
  csharp: ['csharp', 'cs', 'c#'],
  css: ['css'],
  dart: ['dart'],
  haskell: ['haskell', 'hs'],
  html: ['html', 'htm'],
  java: ['java'],
  jsp: ['jsp'],
  javascript: ['javascript', 'js', 'jsx', 'mjs', 'cjs'],
  php: ['php'],
  python: ['python', 'py', 'python3'],
  ruby: ['ruby', 'rb'],
  rust: ['rust', 'rs'],
  typescript: ['typescript', 'ts', 'tsx', 'mts', 'cts'],
  zig: ['zig'],
  text: ['text', 'txt', 'plain', 'plaintext'],
};

const FENCE_LABEL_IDS = new Map<string, number>();
for (const [name, aliases] of Object.entries(FENCE_LABEL_ALIASES)) {
  for (const alias of aliases) FENCE_LABEL_IDS.set(alias, LANGUAGE_IDS[name]!);
}
const MAX_LABEL_LENGTH = Math.max(...[...FENCE_LABEL_IDS.keys()].map((label) => label.length));

/** Fence-scanner state advanced one completed line at a time. */
interface FenceState {
  /** Backtick count of the open fence, or 0 when outside any block. */
  openFenceLength: number;
  /** Language id active inside the open block, or -1 to keep the frame language. */
  blockLanguageId: number;
}

/**
 * Processes one completed line (bytes[start, end), `end` at the terminating LF) against the
 * normative fence grammar. A fence line starts with three or more backticks at column 0;
 * trailing spaces/tabs/CR and leading spaces/tabs of the info string are ignored. Outside a
 * block, an info string containing a backtick disqualifies the line (CommonMark); the label
 * is the first whitespace-delimited word. Inside a block, only a line of at least as many
 * backticks with an empty info string closes it — everything else is content.
 */
function processFenceLine(bytes: Uint8Array, start: number, end: number, state: FenceState): void {
  if (end - start < 3 || bytes[start] !== BACKTICK) return;
  let fenceEnd = start;
  while (fenceEnd < end && bytes[fenceEnd] === BACKTICK) fenceEnd++;
  const fenceLength = fenceEnd - start;
  if (fenceLength < 3) return;
  let restEnd = end;
  while (restEnd > fenceEnd && isTrailingSpace(bytes[restEnd - 1]!)) restEnd--;
  let restStart = fenceEnd;
  while (restStart < restEnd && isSpaceTab(bytes[restStart]!)) restStart++;
  if (state.openFenceLength > 0) {
    if (fenceLength >= state.openFenceLength && restStart === restEnd) {
      state.openFenceLength = 0;
      state.blockLanguageId = -1;
    }
    return;
  }
  for (let i = restStart; i < restEnd; i++) if (bytes[i] === BACKTICK) return;
  let labelEnd = restStart;
  while (labelEnd < restEnd && !isSpaceTab(bytes[labelEnd]!)) labelEnd++;
  state.openFenceLength = fenceLength;
  state.blockLanguageId = resolveLabel(bytes, restStart, labelEnd);
}

/** Trailing trim also strips CR so CRLF line endings need no special casing (normative). */
function isTrailingSpace(byte: number): boolean {
  return byte === 0x20 || byte === 0x09 || byte === 0x0D;
}

/** Leading trim and label termination admit only space/tab — a lone CR is label content. */
function isSpaceTab(byte: number): boolean {
  return byte === 0x20 || byte === 0x09;
}

/** Maps a raw label to its language id, or -1 for unknown labels (keep the frame language). */
function resolveLabel(bytes: Uint8Array, start: number, end: number): number {
  const length = end - start;
  if (length === 0 || length > MAX_LABEL_LENGTH) return -1;
  let label = '';
  for (let i = start; i < end; i++) {
    let code = bytes[i]!;
    if (code >= 0x41 && code <= 0x5A) code += 32;
    if (code > 0x7E) return -1;
    label += String.fromCodePoint(code);
  }
  return FENCE_LABEL_IDS.get(label) ?? -1;
}

/**
 * Encoder policy (not normative): a block language must cover at least this many input
 * bytes — fence lines excluded — before its extension index is built. Building and
 * process-caching a hash index over a ~1 MB dictionary costs milliseconds and megabytes;
 * tiny blocks cannot repay that, while for real blocks (median ~122 bytes in the bench
 * corpora) the cache amortizes across the process like the frame-language index does.
 */
const MIN_EXTENSION_CONTENT = 64;

/**
 * Encoder policy (not normative): at most this many distinct block languages extend one
 * document (the largest by content win; ties break by language id for determinism). This
 * bounds worst-case index memory for pathological many-language documents.
 */
const MAX_EXTENSION_LANGUAGES = 4;

/**
 * Encoder pre-scan: splits the input into dictionary segments at fence transitions.
 * Returns undefined when no position has an extension (the common case, decided by a cheap
 * backtick scan first). A block language that is unknown, unregistered, the frame language
 * itself, or below {@link MIN_EXTENSION_CONTENT} total content yields no extension — such
 * regions match against the frame dictionary exactly like plain v2, and their frames need
 * no registration to decode.
 */
export function computeDictSegments(
  bytes: Uint8Array,
  language: RegisteredLanguage,
  maxDictStart: number
): DictSegment[] | undefined {
  // Extension offsets start at the frame dictionary length; when the shipped mode cannot
  // represent even the first one, indexing block languages would be pure waste.
  if (language.dictionary.length >= maxDictStart) return undefined;
  if (!bytes.includes(BACKTICK)) return undefined;
  const state: FenceState = { openFenceLength: 0, blockLanguageId: -1 };
  // Phase 1: collect fence transitions and per-language content sizes without touching any
  // dictionary index, so unhelpful fences cost only the line scan.
  let starts: number[] | undefined;
  let blocks: (RegisteredLanguage | undefined)[] | undefined;
  const contentBytes = new Map<RegisteredLanguage, number>();
  let currentBlock: RegisteredLanguage | undefined;
  let currentStart = 0;
  let lineStart = 0;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] !== LF) continue;
    const thisLineStart = lineStart;
    processFenceLine(bytes, lineStart, i, state);
    lineStart = i + 1;
    const id = state.blockLanguageId;
    const block = id >= 0 && id !== language.id ? languageById(id) : undefined;
    if (block === currentBlock) continue;
    // A block ends at its closing fence line, which is not content (threshold accounting).
    if (currentBlock) {
      contentBytes.set(currentBlock, (contentBytes.get(currentBlock) ?? 0) + thisLineStart - currentStart);
    }
    starts ??= [0];
    blocks ??= [undefined];
    starts.push(lineStart);
    blocks.push(block);
    currentBlock = block;
    currentStart = lineStart;
  }
  if (!starts || !blocks) return undefined;
  if (currentBlock) {
    contentBytes.set(currentBlock, (contentBytes.get(currentBlock) ?? 0) + bytes.length - currentStart);
  }
  // Phase 2: index only languages whose blocks can repay it, largest content first.
  const qualifying = new Set(
    [...contentBytes.entries()]
      .filter(([, content]) => content >= MIN_EXTENSION_CONTENT)
      .toSorted(([a, ca], [b, cb]) => cb - ca || a.id - b.id)
      .slice(0, MAX_EXTENSION_LANGUAGES)
      .map(([block]) => block)
  );
  if (qualifying.size === 0) return undefined;
  const segments = starts.map((start, i): DictSegment => {
    const block = blocks[i];
    if (!block || !qualifying.has(block)) {
      return { start, extDictionary: undefined, extIndex: undefined, extBase: 0, extWrapperLength: 0 };
    }
    return {
      start,
      extDictionary: block.dictionary,
      extIndex: dictIndexFor(block),
      extBase: language.dictionary.length - block.wrapperLength,
      extWrapperLength: block.wrapperLength,
    };
  });
  return segments;
}

/**
 * True when a dictionary token addresses the extended space above the frame dictionary —
 * the normative condition for setting FLAG_FENCED, so frames whose matches all stay inside
 * the frame dictionary remain bit-identical to plain v2 frames.
 */
export function usesExtendedDictionary(tokens: Token[], frameDictionaryLength: number): boolean {
  return tokens.some((token) => token.type === 'dict' && token.start + token.len > frameDictionaryLength);
}

/**
 * Decode-side copy of a dictionary match that reaches above the frame dictionary: the
 * extension bytes come from the active fence language's dictionary suffix. Throws when no
 * fence language is active at `produced` or the match leaves the extended space.
 */
export function copyExtendedDictMatch(
  out: Uint8Array,
  produced: number,
  start: number,
  length: number,
  frame: RegisteredLanguage,
  tracker: FenceTracker
): void {
  const frameLength = frame.dictionary.length;
  const languageId = tracker.languageIdAt(out, produced);
  if (languageId === frame.id) throw new TokzipDecodeError('dictionary match out of bounds');
  const block = requireLanguageById(languageId);
  if (start + length > frameLength + (block.dictionary.length - block.wrapperLength)) {
    throw new TokzipDecodeError('dictionary match out of bounds');
  }
  // The match may straddle the frame/extension boundary (contiguous virtual space).
  const fromFrame = start < frameLength ? frameLength - start : 0;
  if (fromFrame > 0) out.set(frame.dictionary.subarray(start, frameLength), produced);
  const extStart = start + fromFrame - frameLength + block.wrapperLength;
  out.set(block.dictionary.subarray(extStart, extStart + length - fromFrame), produced + fromFrame);
}

/**
 * Incremental decoder-side fence tracker over the produced output (FLAG_FENCED frames).
 * Bytes after the last produced LF are pending and never affect state, mirroring the
 * encoder's transition points at each LF + 1.
 */
export class FenceTracker {
  private readonly frameId: number;
  private readonly state: FenceState = { openFenceLength: 0, blockLanguageId: -1 };
  private scanned = 0;
  private lineStart = 0;

  constructor(frameId: number) {
    this.frameId = frameId;
  }

  /** Language id addressed by a dictionary match starting at `produced`. */
  languageIdAt(out: Uint8Array, produced: number): number {
    for (let i = this.scanned; i < produced; i++) {
      if (out[i] === LF) {
        processFenceLine(out, this.lineStart, i, this.state);
        this.lineStart = i + 1;
      }
    }
    this.scanned = produced;
    return this.state.blockLanguageId >= 0 ? this.state.blockLanguageId : this.frameId;
  }
}
