import type { DictMatcher } from './dictMatcher.ts';
import { TokzipDecodeError } from './errors.ts';
import { LIT_CLASS_MAX, modelSizeFor } from './format.ts';
import { PROB_SCALE } from './rangeCoder.ts';

/**
 * A language's trained coding model: the literal context classes and the initial 11-bit
 * probabilities for every adaptive model node (layout in format.ts). Model data is codec
 * identity: both sides must run the same priors or decoding diverges.
 */
export interface LanguageModel {
  /** Trained literal context class per previous-byte value (256 entries, values < litClassCount). */
  litContext: Uint8Array;
  /** Number of trained literal context classes (1–{@link LIT_CLASS_MAX}). */
  litClassCount: number;
  /** Initial probabilities, P(bit = 0) in [1, 2047], laid out per format.ts. */
  priors: Uint16Array;
}

/** Data shipped by a language module (or by core for id 0). */
export interface LanguageModuleData {
  id: number;
  name: string;
  /** Language-specific dictionary suffix, appended after the shared wrapper dictionary. */
  dictionarySuffix: Uint8Array;
  model: LanguageModel;
}

/** A registered language with its assembled dictionary and lazily built match indexes. */
export interface RegisteredLanguage {
  id: number;
  name: string;
  /** Wrapper dictionary + language suffix, contiguous. */
  dictionary: Uint8Array;
  /** Byte length of the shared wrapper prefix inside {@link dictionary}. */
  wrapperLength: number;
  model: LanguageModel;
  /** Lazily built hash index over the dictionary (see lz.ts); cached per process. */
  dictIndex: DictIndex | undefined;
  /** Lazily built suffix-automaton matcher (see dictMatcher.ts); cached per process. */
  dictMatcher: DictMatcher | undefined;
}

export interface DictIndex {
  hashShift: number;
  /** 4-byte-hash chain heads: lowest dictionary position per bucket (chains ascend), -1 for empty. */
  head: Int32Array;
  /** Next higher position with the same 4-byte hash, per dictionary position. */
  prev: Int32Array;
  /**
   * 6-byte-hash chains: far more selective on the large repetitive preset dictionaries, so
   * the greedy parse can search deep for long matches without walking useless candidates.
   */
  head6: Int32Array;
  prev6: Int32Array;
}

const byId = new Map<number, RegisteredLanguage>();
const byName = new Map<string, RegisteredLanguage>();

/**
 * Registers a language module. Called by module side-effect imports; validates the model at
 * registration. Re-registering byte-identical module data under the same id/name is a
 * no-op; any diverging registration is rejected (module data is codec identity).
 */
export function registerLanguage(wrapperDictionary: Uint8Array, data: LanguageModuleData): void {
  if (!Number.isInteger(data.id) || data.id < 0 || data.id > 63)
    throw new RangeError(`invalid language id: ${data.id}`);
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
  validateModel(data.model);
  const dictionary = new Uint8Array(wrapperDictionary.length + data.dictionarySuffix.length);
  dictionary.set(wrapperDictionary, 0);
  dictionary.set(data.dictionarySuffix, wrapperDictionary.length);
  const registered: RegisteredLanguage = {
    id: data.id,
    name: data.name,
    dictionary,
    wrapperLength: wrapperDictionary.length,
    // Private copies: callers keep their arrays, so later mutation cannot corrupt the codec.
    model: {
      litContext: new Uint8Array(data.model.litContext),
      litClassCount: data.model.litClassCount,
      priors: new Uint16Array(data.model.priors),
    },
    dictIndex: undefined,
    dictMatcher: undefined,
  };
  byId.set(data.id, registered);
  byName.set(data.name, registered);
}

function sameModuleData(existing: RegisteredLanguage, data: LanguageModuleData): boolean {
  return (
    equalBytes(existing.dictionary.subarray(existing.wrapperLength), data.dictionarySuffix) &&
    existing.model.litClassCount === data.model.litClassCount &&
    equalBytes(existing.model.litContext, data.model.litContext) &&
    equalArrays(existing.model.priors, data.model.priors)
  );
}

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function equalArrays(a: Uint16Array, b: Uint16Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function languageByName(name: string): RegisteredLanguage | undefined {
  return byName.get(name);
}

/**
 * Drops the lazily built per-language match indexes (hash chains and suffix-automaton
 * matchers, ~1 MB per language at the default dictionary budget). They rebuild
 * transparently on the next compress, so this is purely a memory-pressure lever for
 * long-lived processes that compressed in many languages; frames are unaffected.
 */
export function releaseLanguageIndexes(name?: string): void {
  const targets = name === undefined ? byName.values() : ([byName.get(name)].filter(Boolean) as RegisteredLanguage[]);
  for (const language of targets) {
    language.dictIndex = undefined;
    language.dictMatcher = undefined;
  }
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

function validateModel(model: LanguageModel): void {
  const { litClassCount } = model;
  if (!Number.isInteger(litClassCount) || litClassCount < 1 || litClassCount > LIT_CLASS_MAX) {
    throw new RangeError(`invalid literal class count: ${litClassCount}`);
  }
  if (model.litContext.length !== 256) throw new RangeError('literal context map must have 256 entries');
  for (const cls of model.litContext) {
    if (cls >= litClassCount) throw new RangeError('literal context class out of range');
  }
  if (model.priors.length !== modelSizeFor(litClassCount)) {
    throw new RangeError('model priors have wrong length');
  }
  for (const p of model.priors) {
    if (p < 1 || p > PROB_SCALE - 1) throw new RangeError('model prior out of range');
  }
}
