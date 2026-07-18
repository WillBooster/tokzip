import { id0Module } from './generated/core.ts';
import { registerLanguageModule } from './moduleRegistry.ts';

// The dictionary-less `none` path (id 0: wrapper dictionary + generic tables) ships in core.
registerLanguageModule(id0Module);

export { compress, decompress, type CompressOptions, type DecompressOptions } from './container.ts';
export { compressForStorage } from './storage.ts';
export { inspectFrame, type FrameInfo } from './validator.ts';
export {
  TokzipCompressionStream,
  TokzipDecompressionStream,
  type CompressionStreamOptions,
  type DecompressionStreamOptions,
} from './stream.ts';
export { TokzipDecodeError } from './errors.ts';
export { registerLanguageModule } from './moduleRegistry.ts';
export type { LanguageModuleData } from './dictionary.ts';
export { LANGUAGE_IDS } from './languageIds.ts';
