# tokzip

[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![Benchmark](https://github.com/WillBooster/tokzip/actions/workflows/benchmark.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/benchmark.yml)

Lossless compressor specialized for **source code and natural-language text** — human-written
or LLM-generated. Pure TypeScript (no WASM, no native deps), runs in Node/Bun/browsers, and
emits **safe-ASCII text directly** (JSON- and template-literal-safe; `fast` frames are also
URL-safe, while `small` bodies use a radix-85 alphabet that needs percent-encoding inside
URLs) instead of paying the 33% base64 tax on a binary stream.

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

## Benchmarks

**Live dashboard (per-commit charts): <https://willbooster.github.io/tokzip/>**

Every push to `main` runs the [Benchmark workflow](.github/workflows/benchmark.yml) on the
frozen `bench-v1` corpus split (1,848 documents, ~12 MB, sampled from pinned OSS repos and
natural-language sources across 21 languages/locales). The run measures output size on the
_text channel_ (competitors pay the base64 tax on their binary output; tokzip emits safe
ASCII directly), compress/decompress throughput, and **verifies every document round-trips
losslessly in both modes** — a mismatch fails the run.

Representative results (output/input, lower is better; Apple Silicon, Bun 1.3):

| corpus       | docs | tokzip fast | tokzip small | b64(brotli q11) | b64(gzip -6) | b64(zstd -19) |
| ------------ | ---: | ----------: | -----------: | --------------: | -----------: | ------------: |
| typescript   |   29 |       35.8% |        26.8% |           29.0% |        32.9% |         32.1% |
| javascript   |  150 |       37.3% |        28.4% |           27.4% |        32.3% |         31.4% |
| python       |    7 |       43.1% |        32.0% |           29.8% |        34.7% |         33.3% |
| java         |  195 |       31.5% |        23.8% |           27.6% |        34.4% |         33.9% |
| csharp       |  141 |       27.2% |        20.3% |           21.9% |        26.0% |         25.6% |
| rust         |   53 |       31.8% |        24.1% |           23.9% |        27.4% |         26.6% |
| en-US        |  245 |       67.0% |        51.5% |           44.1% |        55.1% |         53.1% |
| ja-JP        |  129 |       65.7% |        51.8% |           46.9% |        53.5% |         50.6% |
| **all (21)** | 1848 |   **56.1%** |    **43.8%** |       **40.3%** |    **48.1%** |     **45.7%** |

tokzip is decisive on short inputs, where general-purpose compressors have nothing to work
with: aggregated over the ≤1 KiB bucket alone, tokzip small emits **53.4%** vs brotli's
62.5% and gzip's 79.8% (≤4 KiB bucket: 45.1% vs 46.7% / 57.6%). Throughput over the whole
corpus: `fast` compresses at ~14 MB/s and decompresses at ~132 MB/s; `small` at ~4.6 MB/s
and ~62 MB/s.

```bash
bun scripts/bench/bench.ts                      # size table + round-trip verification
bun scripts/bench/bench.ts --speed --json out.json  # + MB/s and a machine-readable report
```

## Development

```bash
bun test                                    # round-trip + conformance vectors
bun scripts/corpus/fetchOss.ts --quick      # fetch the OSS code corpus (git-ignored)
bun scripts/corpus/fetchNl.ts               # fetch the natural-language corpus
bun scripts/corpus/generateLlm.ts --limit 8 # generate LLM-written samples (subagent skill)
bun scripts/corpus/split.ts                 # seeded train/bench split (bench-v1)
bun scripts/train/train.ts --all            # train dictionaries + tables → src/generated/
bun scripts/bench/bench.ts                  # size vs base64(brotli/zstd/gzip) + round-trip check; --speed, --json
```

Corpus data is never committed; the scripts, source manifests, and trained modules are.
