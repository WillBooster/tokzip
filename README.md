# tokzip

[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![Benchmark](https://github.com/WillBooster/tokzip/actions/workflows/benchmark.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/benchmark.yml)

Lossless compressor specialized for **source code and natural-language text** — human-written
or LLM-generated. Pure TypeScript (no WASM, no native deps), runs in Node/Bun/browsers, and
emits either of two output channels: **safe-ASCII text directly** (JSON- and
template-literal-safe; `fast` frames are also URL-safe, while `small` bodies use a radix-85
alphabet that needs percent-encoding inside URLs) instead of paying the 33% base64 tax on a
binary stream, or a **dense binary frame** (the same streams packed at 8 bits per byte —
about 25% smaller than the text frame for `fast`, 20% for `small`) for transports that
accept raw bytes.

```ts
import { compress, decompress } from './src/index.ts';
import './src/languages/typescript.ts'; // Self-registers the TypeScript dictionary.

const packed = compress(source, { language: 'typescript', mode: 'small' });
const restored = decompress(packed); // === source

const bytes = compress(source, { language: 'typescript', mode: 'small', output: 'binary' });
const restored2 = decompress(bytes); // === source (Uint8Array in, text/bytes out per frame)
```

- Exactly **two modes**: `fast` (speed-first: greedy parse into a char-aligned radix-64
  stream) and `small` (size-first: an exact-bit-price optimal parse feeding context-modeled
  static entropy coding — literals keyed by trained previous-byte classes, token symbols by
  the previous token kind, offsets by match kind — through a fused radix-85 writer, with
  normative auto-downgrade so output never expands beyond a stored frame).
- **Per-language preset dictionaries** (17 programming languages + 4 locales, tree-shakeable
  modules of up to 1 MB dictionary each — the full `small`-mode offset addressing range)
  plus a shared wrapper dictionary in core — decisive on short inputs where general-purpose
  compressors have nothing to work with.
- **Fence-aware dictionary extension**: inside a labeled triple-backtick code block
  (` ```ts `, ` ```python `, …) the searchable dictionary space automatically grows by that
  language's dictionary — Markdown docs and LLM output with embedded code get both the
  surrounding document's dictionary and the right code dictionary per block. Unlabeled or
  unknown labels keep the plain space; a block language's module must be registered on both
  sides only when a match actually uses its dictionary (see FORMAT.md §6.1).
- Never fails on malformed/partial input; corrupt payloads throw a typed `TokzipDecodeError`.

The wire format is specified in [FORMAT.md](FORMAT.md); the design rationale lives in
[issue #2](https://github.com/WillBooster/tokzip/issues/2).

## Streaming

`TokzipCompressionStream` / `TokzipDecompressionStream` are Web Streams
(`TransformStream<Uint8Array | string, Uint8Array>`), so the same code pipes in Node.js 18+
and browsers — mirroring the built-in `CompressionStream` API. The whole mechanism is hidden
inside the stream object: input is cut into blocks (256 KB by default), the LZ window is
carried across block boundaries, and every block independently ships the smallest of
stored/fast/small bodies.

```ts
import { TokzipCompressionStream, TokzipDecompressionStream } from './src/index.ts';
import './src/languages/typescript.ts';

const compressed = readable.pipeThrough(new TokzipCompressionStream({ language: 'typescript' }));
const restored = compressed.pipeThrough(new TokzipDecompressionStream());
```

Streams use their own block container (binary channel only) and reach one-shot-or-better
ratios: with default options, `mode: 'fast'` benches 1–2% _smaller_ than one-shot `fast`
(streams enable price-aware lazy matching), and `mode: 'small'` benches 3–7% smaller than
one-shot `small` on multi-megabyte inputs, whose blocks stay inside the optimal parser's
input bound while one-shot compression falls back to the greedy parse. Memory stays
O(blockSize + window) on both sides regardless of stream length. Options: `blockSize` trades
latency/memory for ratio in `fast` mode (in `small` mode the 256 KB default is the practical
ceiling — larger blocks shrink the history budget and past 512 KB lose the optimal parse),
`carryWindow: false` makes blocks independently decodable, and `historyLimit` bounds the
carried window (compression-speed lever for small blocks); run
`bun scripts/bench/streamBench.ts` (add `--history` for the `historyLimit` sweep) to see the
trade-offs on the seeded corpus.

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

The harness measures two channels separately. The **text channel** (default) compares
URL-safe text output: binary codecs use unpadded base64url, while tokzip and the `lz-string`
URI mode already emit text. The **binary channel** (`--binary`) compares tokzip binary
frames against the raw codec bytes with no text framing. Both channels measure median
end-to-end per-document throughput and **verify every method on every document round-trips
losslessly** — any mismatch fails the run.

Reference run, pinned at corpus fingerprint `5a25e65df399` (`bench-v2`, 2,493 documents,
~11.1 MB, 21 languages/locales). Ratio is output/input per channel (lower is better;
identical across machines for a given corpus); speed is median end-to-end per-document
throughput with the pinned Bun on the reference machine of this run (the live dashboard
tracks per-commit speeds on a standard GitHub Actions runner):

**Text channel** (URL-safe output; binary codecs pay unpadded base64url):

| method             | output / input |   compress | decompress |
| ------------------ | -------------: | ---------: | ---------: |
| **tokzip small**   |      **26.5%** |   2.0 MB/s | 140.8 MB/s |
| b64url(brotli q11) |          34.4% |   1.1 MB/s | 197.5 MB/s |
| b64url(brotli q5)  |          38.7% |  58.0 MB/s | 223.0 MB/s |
| **tokzip fast**    |      **40.1%** |  31.9 MB/s | 190.8 MB/s |
| b64url(zstd -19)   |          40.7% |   5.9 MB/s | 361.4 MB/s |
| b64url(gzip -6)    |          41.9% |  97.9 MB/s | 258.7 MB/s |
| b64url(xz -9e)     |          42.3% |          — |          — |
| b64url(zstd -3)    |          44.1% | 227.5 MB/s | 354.7 MB/s |
| lz-string URI      |          61.6% |   8.6 MB/s |  22.9 MB/s |

**Binary channel** (tokzip binary frames vs the raw codec bytes, no text framing):

| method           | output / input |   compress | decompress |
| ---------------- | -------------: | ---------: | ---------: |
| **tokzip small** |      **21.2%** |   2.0 MB/s | 127.1 MB/s |
| brotli q11       |          25.8% |   1.1 MB/s | 258.8 MB/s |
| brotli q5        |          29.0% |  58.2 MB/s | 291.7 MB/s |
| **tokzip fast**  |      **30.1%** |  31.8 MB/s | 169.0 MB/s |
| zstd -19         |          30.5% |   5.9 MB/s | 594.6 MB/s |
| gzip -6          |          31.4% |  99.2 MB/s | 372.4 MB/s |
| xz -9e           |          31.7% |          — |          — |
| zstd -3          |          33.1% | 248.2 MB/s | 637.9 MB/s |

`small` produces the smallest output of every measured method on both channels — 7.9
points below brotli q11 on the text channel and 4.6 points below raw brotli q11 on the
binary channel, while compressing about twice as fast as it. `fast` beats zstd -19's ratio
on both channels at over five times its compression speed while staying URL-safe as text.
(The xz -9e reference runs through the system CLI, so it is measured for size and
round-trip only.)

On the private `tokzip-corpus-private` evaluation set (226 sanitized production LLM
outputs, ~2.4 MB across css/html/text/typescript, fingerprint `27229c766758`) tokzip
small still leads the text channel (28.7% vs brotli q11's 28.9%), while raw brotli q11
leads the binary channel (21.7% vs 22.9%): those documents average ~11 KB, where preset
dictionaries matter less than brotli's larger backward window. Dictionaries are trained
exclusively on the public corpus — private production content never flows into them.

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
bun scripts/bench/bench.ts                      # text channel: size table + round-trip verification
bun scripts/bench/bench.ts --binary             # binary channel: tokzip binary frames vs raw codec bytes
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
