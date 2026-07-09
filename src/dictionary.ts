import { OFFSET_SLOT_COUNT } from './slots.ts';
import { TOKEN_ALPHABET_SIZE } from './format.ts';
import { isCompleteCode } from './huffman.ts';
import { TokzipDecodeError } from './errors.ts';

/** Static `small`-mode canonical Huffman code lengths for the three separated streams. */
export interface EntropyTables {
  literal: Uint8Array; // 256 symbols
  token: Uint8Array; // TOKEN_ALPHABET_SIZE symbols
  offset: Uint8Array; // OFFSET_SLOT_COUNT symbols
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
  top64: Uint8Array;
  /** Maps byte value → literal-64 index, or -1 when the byte is outside the charset. */
  top64Index: Int8Array;
  tables: EntropyTables;
  /** Lazily built hash index over the dictionary (see lz.ts); cached per process. */
  dictIndex: DictIndex | undefined;
}

export interface DictIndex {
  hashShift: number;
  /** Bucketed positions (4 slots per bucket), -1 for empty. */
  table: Int32Array;
}

const byId = new Map<number, RegisteredLanguage>();
const byName = new Map<string, RegisteredLanguage>();

/**
 * Registers a language module. Called by module side-effect imports; validates tables at
 * registration and is idempotent (re-registering the same id/name replaces the entry).
 */
export function registerLanguage(wrapperDictionary: Uint8Array, data: LanguageModuleData): void {
  if (!Number.isInteger(data.id) || data.id < 0 || data.id > 63)
    throw new RangeError(`invalid language id: ${data.id}`);
  if (data.top64.length !== 64) throw new RangeError('top-64 charset must contain exactly 64 bytes');
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
    top64: data.top64,
    top64Index,
    tables: data.tables,
    dictIndex: undefined,
  };
  byId.set(data.id, registered);
  byName.set(data.name, registered);
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
  if (
    tables.literal.length !== 256 ||
    tables.token.length !== TOKEN_ALPHABET_SIZE ||
    tables.offset.length !== OFFSET_SLOT_COUNT
  ) {
    throw new RangeError('entropy table has wrong alphabet size');
  }
  for (const [name, lengths] of [
    ['literal', tables.literal],
    ['token', tables.token],
    ['offset', tables.offset],
  ] as const) {
    if (!isCompleteCode(lengths)) throw new RangeError(`entropy table "${name}" is not a complete code`);
  }
}
