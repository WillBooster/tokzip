import { OFFSET_SLOT_COUNT } from './slots.ts';
import { LIT_CLASS_MAX, OFFSET_CONTEXT_COUNT, TOKEN_ALPHABET_SIZE, TOKEN_CONTEXT_COUNT } from './format.ts';
import { isCompleteCode } from './huffman.ts';
import { TokzipDecodeError } from './errors.ts';

/**
 * Static `small`-mode canonical Huffman code lengths for the three separated streams,
 * one table per context (see format.ts): literals keyed by the trained class of the
 * previous byte, token symbols by the previous token kind, offsets by the match kind.
 */
export interface EntropyTables {
  /** Trained literal context class per previous-byte value (256 entries, values < litClassCount). */
  litContext: Uint8Array;
  /** Number of trained literal context classes (1–{@link LIT_CLASS_MAX}). */
  litClassCount: number;
  literal: Uint8Array; // litClassCount × 256 symbols
  token: Uint8Array; // TOKEN_CONTEXT_COUNT × TOKEN_ALPHABET_SIZE symbols
  offset: Uint8Array; // OFFSET_CONTEXT_COUNT × OFFSET_SLOT_COUNT symbols
}

/** Data shipped by a language module (or by core for id 0). */
export interface LanguageModuleData {
  id: number;
  name: string;
  /** Language-specific dictionary suffix, appended after the shared wrapper dictionary. */
  dictionarySuffix: Uint8Array;
  /** The 64 most frequent literal bytes (the `fast`-mode literal-64 charset). */
  top64: Uint8Array;
  tables: EntropyTables;
}

/** A registered language with its assembled dictionary and lazily built match index. */
export interface RegisteredLanguage {
  id: number;
  name: string;
  /** Wrapper dictionary + language suffix, contiguous. */
  dictionary: Uint8Array;
  /** Byte length of the shared wrapper prefix inside {@link dictionary}. */
  wrapperLength: number;
  top64: Uint8Array;
  /** Maps byte value → literal-64 index, or -1 when the byte is outside the charset. */
  top64Index: Int8Array;
  tables: EntropyTables;
  /** Lazily built hash index over the dictionary (see lz.ts); cached per process. */
  dictIndex: DictIndex | undefined;
}

export interface DictIndex {
  hashShift: number;
  /** 4-byte-hash chain heads: lowest dictionary position per bucket (chains ascend), -1 for empty. */
  head: Int32Array;
  /** Next higher position with the same 4-byte hash, per dictionary position. */
  prev: Int32Array;
  /**
   * 6-byte-hash chains: far more selective on the large repetitive preset dictionaries, so
   * the optimal parse can search deep for long matches without walking useless candidates.
   */
  head6: Int32Array;
  prev6: Int32Array;
}

const byId = new Map<number, RegisteredLanguage>();
const byName = new Map<string, RegisteredLanguage>();

/**
 * Registers a language module. Called by module side-effect imports; validates tables at
 * registration. Re-registering byte-identical module data under the same id/name is a
 * no-op; any diverging registration is rejected (module data is codec identity).
 */
export function registerLanguage(wrapperDictionary: Uint8Array, data: LanguageModuleData): void {
  if (!Number.isInteger(data.id) || data.id < 0 || data.id > 63)
    throw new RangeError(`invalid language id: ${data.id}`);
  if (data.top64.length !== 64) throw new RangeError('top-64 charset must contain exactly 64 bytes');
  // compress selects by name while decompress selects by id: a conflicting registration would
  // let the two maps diverge and silently decode with the wrong dictionary. Re-registering the
  // same (id, name) pair is idempotent only for byte-identical module data — module data is
  // codec identity (FORMAT.md §10), so replacing it would silently invalidate persisted frames.
  const existingById = byId.get(data.id);
  if (existingById && existingById.name !== data.name) {
    throw new RangeError(`language id ${data.id} is already registered as "${existingById.name}"`);
  }
  const existingByName = byName.get(data.name);
  if (existingByName && existingByName.id !== data.id) {
    throw new RangeError(`language "${data.name}" is already registered with id ${existingByName.id}`);
  }
  if (existingById) {
    if (!sameModuleData(existingById, data)) {
      throw new RangeError(`language "${data.name}" is already registered with different module data`);
    }
    return;
  }
  validateTables(data.tables);
  const dictionary = new Uint8Array(wrapperDictionary.length + data.dictionarySuffix.length);
  dictionary.set(wrapperDictionary, 0);
  dictionary.set(data.dictionarySuffix, wrapperDictionary.length);
  const top64Index = new Int8Array(256).fill(-1);
  for (let i = 0; i < 64; i++) {
    const byte = data.top64[i]!;
    if (top64Index[byte] === -1) top64Index[byte] = i;
  }
  const registered: RegisteredLanguage = {
    id: data.id,
    name: data.name,
    dictionary,
    wrapperLength: wrapperDictionary.length,
    // Private copies: callers keep their arrays, so later mutation cannot corrupt the codec.
    top64: new Uint8Array(data.top64),
    top64Index,
    tables: {
      litContext: new Uint8Array(data.tables.litContext),
      litClassCount: data.tables.litClassCount,
      literal: new Uint8Array(data.tables.literal),
      token: new Uint8Array(data.tables.token),
      offset: new Uint8Array(data.tables.offset),
    },
    dictIndex: undefined,
  };
  byId.set(data.id, registered);
  byName.set(data.name, registered);
}

function sameModuleData(existing: RegisteredLanguage, data: LanguageModuleData): boolean {
  return (
    equalBytes(existing.dictionary.subarray(existing.wrapperLength), data.dictionarySuffix) &&
    equalBytes(existing.top64, data.top64) &&
    existing.tables.litClassCount === data.tables.litClassCount &&
    equalBytes(existing.tables.litContext, data.tables.litContext) &&
    equalBytes(existing.tables.literal, data.tables.literal) &&
    equalBytes(existing.tables.token, data.tables.token) &&
    equalBytes(existing.tables.offset, data.tables.offset)
  );
}

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function languageByName(name: string): RegisteredLanguage | undefined {
  return byName.get(name);
}

export function languageById(id: number): RegisteredLanguage | undefined {
  return byId.get(id);
}

/** Decoder-side lookup: unknown ids are a structural error for non-stored frames. */
export function requireLanguageById(id: number): RegisteredLanguage {
  const language = byId.get(id);
  if (!language) throw new TokzipDecodeError(`unknown language id: ${id}`);
  return language;
}

function validateTables(tables: EntropyTables): void {
  const { litClassCount } = tables;
  if (!Number.isInteger(litClassCount) || litClassCount < 1 || litClassCount > LIT_CLASS_MAX) {
    throw new RangeError(`invalid literal class count: ${litClassCount}`);
  }
  if (tables.litContext.length !== 256) throw new RangeError('literal context map must have 256 entries');
  for (const cls of tables.litContext) {
    if (cls >= litClassCount) throw new RangeError('literal context class out of range');
  }
  if (
    tables.literal.length !== litClassCount * 256 ||
    tables.token.length !== TOKEN_CONTEXT_COUNT * TOKEN_ALPHABET_SIZE ||
    tables.offset.length !== OFFSET_CONTEXT_COUNT * OFFSET_SLOT_COUNT
  ) {
    throw new RangeError('entropy table has wrong alphabet size');
  }
  for (const [name, lengths, alphabetSize] of [
    ['literal', tables.literal, 256],
    ['token', tables.token, TOKEN_ALPHABET_SIZE],
    ['offset', tables.offset, OFFSET_SLOT_COUNT],
  ] as const) {
    for (let base = 0; base < lengths.length; base += alphabetSize) {
      if (!isCompleteCode(lengths.subarray(base, base + alphabetSize))) {
        throw new RangeError(`entropy table "${name}" context ${base / alphabetSize} is not a complete code`);
      }
    }
  }
}
