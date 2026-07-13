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

- Exactly **two modes**: `fast` (speed-first: greedy parse into a char-aligned radix-64
  stream) and `small` (size-first: an exact-bit-price optimal parse feeding context-modeled
  static entropy coding — literals keyed by trained previous-byte classes, token symbols by
  the previous token kind, offsets by match kind — through a fused radix-85 writer, with
  normative auto-downgrade so output never expands beyond a stored frame).
- **Per-language preset dictionaries** (17 programming languages + 4 locales, tree-shakeable
  modules of up to 512 KB dictionary each) plus a shared wrapper dictionary in core —
  decisive on short inputs where general-purpose compressors have nothing to work with.
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

Latest local run (`bench-v2`, fingerprint `5a25e65df399`, 2,493 documents, ~11.1 MB,
21 languages/locales; output/input, lower is better; Apple Silicon, Bun 1.3):

| corpus       | docs | tokzip fast | tokzip small | b64url(brotli q11) | b64url(gzip -6) | b64url(zstd -19) | b64url(xz -9e) |
| ------------ | ---: | ----------: | -----------: | -----------------: | --------------: | ---------------: | -------------: |
| typescript   |  146 |       28.2% |        18.9% |              22.8% |           26.6% |            25.6% |          27.1% |
| javascript   |   62 |       44.0% |        30.3% |              38.4% |           45.3% |            44.3% |          48.1% |
| python       |   70 |       33.9% |        23.0% |              28.0% |           32.3% |            31.4% |          32.9% |
| java         |   90 |       26.7% |        17.8% |              32.3% |           40.0% |            39.7% |          42.2% |
| csharp       |  231 |       25.7% |        17.3% |              25.1% |           29.7% |            29.3% |          31.9% |
| rust         |   75 |       28.6% |        19.4% |              26.8% |           30.5% |            29.6% |          32.1% |
| en-US        |  347 |       51.8% |        36.2% |              43.7% |           54.9% |            54.1% |          58.2% |
| ja-JP        |  139 |       50.4% |        33.3% |              46.7% |           57.1% |            55.9% |          55.5% |
| **all (21)** | 2493 |   **40.1%** |    **27.1%** |          **34.4%** |       **41.9%** |        **40.7%** |      **42.3%** |

On this per-document workload, tokzip `fast` compresses/decompresses at 37.0/221.0 MB/s
(8.3/49.6 thousand documents/s), and `small` at 3.2/158.4 MB/s. `small` produces the
smallest output of every measured method on **every language and locale** — 7.3 points
below brotli q11 overall while compressing ~2.7× faster than it (brotli q11: 1.2 MB/s),
and 15 points below xz -9e. `fast` beats zstd -19's ratio at five times its compression
speed, and zstd -3 reaches 267.9 MB/s but emits 40.7% of the input size after base64url
framing. (The xz -9e reference runs through the system CLI, so it is measured for size
and round-trip only.)

```bash
bun scripts/bench/bench.ts                      # size table + round-trip verification
bun scripts/bench/bench.ts --speed --json out.json  # + MB/s and a machine-readable report
```

## Development

```bash
bun test                                    # round-trip + conformance vectors
bun scripts/train/train.ts --all            # train dictionaries + tables → src/generated/
bun scripts/bench/bench.ts                  # size vs base64url(brotli/zstd/gzip/xz), lz-string + round-trip; --speed, --json
```

By default, training and benchmarks read `../tokzip-corpus/corpus`, and benchmarks also
detect a sibling `../tokzip-corpus-private` checkout automatically (freshened with
`git pull`) and merge its bench split in. Training never reads the private corpus:
generated dictionaries embed literal training fragments and are committed to this public
repository. Set `TOKZIP_CORPUS_DIR` to use exactly one corpus checkout instead — it
disables the private-corpus detection. Corpus acquisition,
generation, provenance, validation, and splitting live in the dedicated corpus repository;
this repository commits only trained modules and the codec that consumes them.
