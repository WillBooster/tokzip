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

**Live dashboard (per-commit charts, per-language and per-size tables):
<https://willbooster.github.io/tokzip/>**

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

Reference run, pinned at corpus fingerprint `5a25e65df399` (`bench-v2`, 2,493 documents,
~11.1 MB, 21 languages/locales). Ratio is text-channel output/input (lower is better;
identical across machines for a given corpus); speed is median end-to-end per-document
throughput on a standard GitHub Actions runner with the pinned Bun:

| method             | output / input |   compress | decompress |
| ------------------ | -------------: | ---------: | ---------: |
| **tokzip small**   |      **27.1%** |   1.9 MB/s |  95.1 MB/s |
| b64url(brotli q11) |          34.4% |   0.9 MB/s | 152.6 MB/s |
| b64url(brotli q5)  |          38.7% |  41.6 MB/s | 166.0 MB/s |
| **tokzip fast**    |      **40.1%** |  19.2 MB/s | 127.1 MB/s |
| b64url(zstd -19)   |          40.7% |   4.9 MB/s | 256.4 MB/s |
| b64url(gzip -6)    |          41.9% |  75.8 MB/s | 204.9 MB/s |
| b64url(xz -9e)     |          42.3% |          — |          — |
| b64url(zstd -3)    |          44.1% | 149.8 MB/s | 247.8 MB/s |
| lz-string URI      |          61.6% |   4.7 MB/s |  15.9 MB/s |

`small` produces the smallest output of every measured method on **every language and
locale** — 7.3 points below brotli q11 overall while compressing about twice as fast as
it, and 15 points below xz -9e. `fast` beats zstd -19's ratio at four times its
compression speed while staying URL-safe. (The xz -9e reference runs through the system
CLI, so it is measured for size and round-trip only.)

The figures below are regenerated from the newest `main` benchmark run on every push, so
they can be newer than the pinned table above:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://willbooster.github.io/tokzip/charts/ratio-speed-dark.svg" />
  <img alt="Scatter chart of compression speed versus output size for tokzip fast, tokzip small, and the baseline codecs on the newest main run" src="https://willbooster.github.io/tokzip/charts/ratio-speed-light.svg" />
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://willbooster.github.io/tokzip/charts/languages-dark.svg" />
  <img alt="Dot plot of per-language compression ratios for tokzip small, tokzip fast, brotli q11, and zstd -19 on the newest main run" src="https://willbooster.github.io/tokzip/charts/languages-light.svg" />
</picture>

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
