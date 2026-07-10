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
seeded `bench-v1` split from a pinned
[`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus) commit (~980 documents
sampled from one pinned, permissively licensed OSS repo per language plus MIT-licensed
natural-language documentation; a language
can end up with no bench-split documents — currently `html`). Every report includes a SHA-256
fingerprint of the exact corpus bytes so results from changed upstream natural-language
content are not mistaken for like-for-like codec changes.

The harness measures URL-safe text output: binary codecs use unpadded base64url, while
tokzip and the `lz-string` URI mode already emit text. It also measures median end-to-end
per-document throughput, including binary-to-text framing, and **verifies every method on
every document round-trips losslessly** — any mismatch fails the run.

Latest local run (`bench-v1`, fingerprint `5c435549ab34`, 980 documents, ~5.0 MB,
20 languages/locales; output/input, lower is better; Apple Silicon, Bun 1.3):

| corpus       | docs | tokzip fast | tokzip small | b64url(brotli q11) | b64url(gzip -6) | b64url(zstd -19) |
| ------------ | ---: | ----------: | -----------: | -----------------: | --------------: | ---------------: |
| typescript   |   29 |       35.8% |        26.8% |              28.9% |           32.8% |            32.1% |
| javascript   |    4 |       45.9% |        34.1% |              31.5% |           36.5% |            34.9% |
| python       |    7 |       43.1% |        32.0% |              29.8% |           34.7% |            33.3% |
| java         |   39 |       39.7% |        29.8% |              31.6% |           38.6% |            37.7% |
| csharp       |  141 |       27.2% |        20.3% |              21.9% |           26.0% |            25.6% |
| rust         |   53 |       31.8% |        24.2% |              23.9% |           27.4% |            26.6% |
| en-US        |   56 |       63.1% |        48.6% |              46.3% |           58.4% |            57.8% |
| ja-JP        |   53 |       61.0% |        48.2% |              48.3% |           57.6% |            56.1% |
| **all (20)** |  980 |   **43.5%** |    **33.4%** |          **33.2%** |       **40.1%** |        **38.9%** |

On this per-document workload, tokzip `fast` compresses/decompresses at 26.0/192.5 MB/s
(5.4/39.9 thousand documents/s), and `small` at 7.7/28.0 MB/s. Brotli q11 reaches the
smallest overall output but compresses at 0.6 MB/s; zstd -3 reaches 108.4 MB/s but emits
42.3% of the input size after base64url framing.

```bash
bun scripts/bench/bench.ts                      # size table + round-trip verification
bun scripts/bench/bench.ts --speed --json out.json  # + MB/s and a machine-readable report
```

## Development

```bash
bun test                                    # round-trip + conformance vectors
bun scripts/train/train.ts --all            # train dictionaries + tables → src/generated/
bun scripts/bench/bench.ts                  # size vs base64url(brotli/zstd/gzip), lz-string + round-trip; --speed, --json
```

By default, training and benchmarks read `../tokzip-corpus/corpus`. Set
`TOKZIP_CORPUS_DIR` to use another public or private corpus checkout. Corpus acquisition,
generation, provenance, validation, and splitting live in the dedicated corpus repository;
this repository commits only trained modules and the codec that consumes them.
