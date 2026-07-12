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
seeded `bench-v2` split from a pinned
[`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus) commit (~2,000 documents
sampled from pinned, permissively licensed OSS repositories plus permissively licensed
natural-language documentation). Every report includes a SHA-256
fingerprint of the exact corpus bytes so results from changed upstream natural-language
content are not mistaken for like-for-like codec changes.

The harness measures URL-safe text output: binary codecs use unpadded base64url, while
tokzip and the `lz-string` URI mode already emit text. It also measures median end-to-end
per-document throughput, including binary-to-text framing, and **verifies every method on
every document round-trips losslessly** — any mismatch fails the run.

Latest local run (`bench-v2`, fingerprint `230e9b02269c`, 2,024 documents, ~9.8 MB,
21 languages/locales; output/input, lower is better; Apple Silicon, Bun 1.3):

| corpus       | docs | tokzip fast | tokzip small | b64url(brotli q11) | b64url(gzip -6) | b64url(zstd -19) |
| ------------ | ---: | ----------: | -----------: | -----------------: | --------------: | ---------------: |
| typescript   |  146 |       32.9% |        24.5% |              23.5% |           27.4% |            26.3% |
| javascript   |   62 |       51.5% |        38.7% |              38.4% |           45.3% |            44.3% |
| python       |   70 |       38.9% |        29.1% |              28.0% |           32.3% |            31.4% |
| java         |   71 |       36.4% |        27.6% |              30.8% |           37.5% |            37.0% |
| csharp       |  229 |       30.4% |        22.9% |              24.6% |           29.0% |            28.5% |
| rust         |   70 |       33.5% |        25.5% |              25.4% |           29.1% |            28.2% |
| en-US        |   95 |       59.1% |        45.5% |              41.0% |           52.2% |            51.6% |
| ja-JP        |   81 |       61.5% |        48.3% |              49.8% |           59.9% |            59.1% |
| **all (21)** | 2024 |   **43.0%** |    **33.1%** |          **31.6%** |       **38.1%** |        **37.1%** |

On this per-document workload, tokzip `fast` compresses/decompresses at 25.8/179.2 MB/s
(5.6/38.8 thousand documents/s), and `small` at 8.7/82.8 MB/s. Brotli q11 reaches the
smallest overall output but compresses at 1.1 MB/s; zstd -3 reaches 244.0 MB/s but emits
40.3% of the input size after base64url framing.

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

By default, training and benchmarks read `../tokzip-corpus/corpus`, and benchmarks also
detect a sibling `../tokzip-corpus-private` checkout automatically (freshened with
`git pull`) and merge its bench split in. Training never reads the private corpus:
generated dictionaries embed literal training fragments and are committed to this public
repository. Set `TOKZIP_CORPUS_DIR` to use exactly one corpus checkout instead — it
disables the private-corpus detection. Corpus acquisition,
generation, provenance, validation, and splitting live in the dedicated corpus repository;
this repository commits only trained modules and the codec that consumes them.
