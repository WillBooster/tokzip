# tokzip

[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)

Lossless compressor specialized for **source code and natural-language text** — human-written
or LLM-generated. Pure TypeScript (no WASM, no native deps), runs in Node/Bun/browsers, and
emits **safe-ASCII text directly** (JSON/URL/template-literal-safe) instead of paying the 33%
base64 tax on a binary stream.

```ts
import { compress, decompress } from './src/index.ts';
import './src/languages/typescript.ts'; // Self-registers the TypeScript dictionary.

const packed = compress(source, { language: 'typescript', mode: 'small' });
const restored = decompress(packed); // === source
```

- Exactly **two modes**: `fast` (speed-first, char-aligned radix-64 stream) and `small`
  (size-first: static entropy coding through a fused radix-85 writer, with normative
  auto-downgrade so output never expands beyond a stored frame).
- **Per-language preset dictionaries** (17 programming languages + 4 locales, tree-shakeable
  modules) plus a shared wrapper dictionary in core — decisive on short inputs where
  general-purpose compressors have nothing to work with.
- Never fails on malformed/partial input; corrupt payloads throw a typed `TokzipDecodeError`.

The wire format is specified in [FORMAT.md](FORMAT.md); the design rationale lives in
[issue #2](https://github.com/WillBooster/tokzip/issues/2).

## Development

```bash
bun test                                    # round-trip + conformance vectors
bun scripts/corpus/fetchOss.ts --quick      # fetch the OSS code corpus (git-ignored)
bun scripts/corpus/fetchNl.ts               # fetch the natural-language corpus
bun scripts/corpus/generateLlm.ts --limit 8 # generate LLM-written samples (subagent skill)
bun scripts/corpus/split.ts                 # seeded train/bench split (bench-v1)
bun scripts/train/train.ts --all            # train dictionaries + tables → src/generated/
bun scripts/bench/bench.ts                  # size vs base64(brotli/zstd/gzip); --speed for MB/s
```

Corpus data is never committed; the scripts, source manifests, and trained modules are.
