/**
 * Stable v1 language-id allocation (normative; see FORMAT.md). Id 0 is the dictionary-less
 * wrapper-only path; XML is deferred but its id is reserved.
 */
export const LANGUAGE_IDS: Readonly<Record<string, number>> = {
  none: 0,
  text: 1,
  c: 2,
  cpp: 3,
  csharp: 4,
  css: 5,
  dart: 6,
  haskell: 7,
  html: 8,
  java: 9,
  jsp: 10,
  javascript: 11,
  php: 12,
  python: 13,
  ruby: 14,
  rust: 15,
  typescript: 16,
  zig: 17,
  'en-US': 18,
  'ja-JP': 19,
  'zh-CN': 20,
  'zh-TW': 21,
  // xml: 22 (deferred, reserved)
};
