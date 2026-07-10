import { registerLanguage, type LanguageModuleData } from './dictionary.ts';
import { wrapperDictionary } from './generated/core.ts';

/**
 * Registers a language module against the shared wrapper dictionary. Generated modules under
 * src/languages/ call this on import (tree-shakeable side-effect registration).
 */
export function registerLanguageModule(data: LanguageModuleData): void {
  registerLanguage(wrapperDictionary, data);
}
