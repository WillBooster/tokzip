# tokzip

[![Test rust](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml)
[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![wbfy](https://img.shields.io/badge/wbfy-18.7.4-1e90ff.svg)](https://github.com/WillBooster/shared/tree/main/packages/wbfy)

Lossless compressor specialized for **source code and natural-language text** — human-written
or LLM-generated. Pure TypeScript (no WASM, no native deps), runs in Node/Bun/browsers, and
emits either of two output channels: **safe-ASCII text directly** (JSON- and
template-literal-safe radix-85; needs percent-encoding inside URLs) instead of paying the
33% base64 tax on a binary stream, or a **dense binary frame** for transports that accept
raw bytes (when both channels ship the same range-coded body, the text frame pays the 25%
radix-85 tax on it; each channel independently downgrades to stored, and headers/padding
make whole-frame ratios vary a little). On the pinned public benchmark corpus (see
Benchmarks below) it outperforms base64url(brotli -q11) on the text
channel in every language and size bucket, and raw brotli -q11 on the binary channel
overall (25.1% vs 27.1%) and in every language except generic prose `text` (0.3 pp behind —
brotli's built-in dictionary is strong on large English-prose documents, and individual
8–24 KB prose/CJK buckets trail by up to ~1.1 pp at the default 16 KB budget; retraining
at 128 KB via `--budget` wins every bucket on both channels).

```ts
import { compress, decompress } from './src/index.ts';
import './src/languages/typescript.ts'; // Self-registers the TypeScript dictionary.

const packed = compress(source, { language: 'typescript' });
const restored = decompress(packed); // === source

const bytes = compress(source, { language: 'typescript', output: 'binary' });
const restored2 = decompress(bytes); // === source (Uint8Array in, text/bytes out per frame)
```

- **One mode**: an optimal LZ parse priced with the trained priors (exact for inputs up to
  512 KB, greedy-lazy beyond; suffix-automaton dictionary matching, rep-offset cache)
  feeding an adaptive binary range coder whose models — literals keyed by
  trained previous-byte classes with LZMA-style matched-literal prediction, match kinds by
  the previous token kind, offsets by length bucket — start from trained per-language
  priors, so short documents get the full benefit of the static statistics while long
  documents adapt to themselves. Normative auto-downgrade: output never expands beyond a
  stored frame.
- **Per-language preset dictionaries** (17 programming languages + 4 locales, tree-shakeable
  modules; default budget 16 KB per language — chosen for the primary storage deployment,
  where the dictionary ships once with the application, documents compress inside request
  handlers on short-lived isolates, and the first-compress index cost matters as much as
  ratio — retrainable from 4 KB up to the full 1 MB offset range via `--budget`; 128 KB
  buys ~1.2 pp of overall text-channel ratio at ~6× the index cost below) plus a
  shared wrapper dictionary in core — decisive on short
  inputs where general-purpose compressors have nothing to work with. Deployments that
  instead download a dictionary per client session should retrain smaller still (the
  session-amortized benchmark, which charges each dictionary's brotli-compressed transfer
  size against tokzip, previously picked 8 KB; see Benchmarks below). The first compress
  per language builds a suffix-automaton matcher over its dictionary (~10 ms and ~1 MB
  retained at the default budget; ~63 ms and ~7 MB at 128 KB), cached for the process;
  long-lived processes that compressed in many languages can drop the caches with
  `releaseLanguageIndexes()` — they rebuild transparently on the next compress.
- **Mandatory content checksum**: every frame carries a CRC-32 of the decompressed content
  (the same integrity guarantee gzip provides), verified before any output is returned.
- **Storage-grade helpers**: `compressForStorage` verifies the frame round-trips to the
  exact input before returning it and falls back to a plain stored frame on any failure
  (`compress` mirrors WHATWG `TextEncoder` for lone surrogates — they encode as U+FFFD —
  while `compressForStorage` rejects such strings up front so stored data is byte-exact);
  `inspectFrame` validates a frame's header/envelope without decompressing — for servers
  that pass client-compressed payloads through to storage untouched.
- **Fence-aware dictionary extension**: inside a labeled triple-backtick code block
  (` ```ts `, ` ```python `, …) the searchable dictionary space automatically grows by that
  language's dictionary — Markdown docs and LLM output with embedded code get both the
  surrounding document's dictionary and the right code dictionary per block. Unlabeled or
  unknown labels keep the plain space; a block language's module must be registered on both
  sides only when a match actually uses its dictionary (see FORMAT.md §6.1).
- Never fails on malformed/partial input; corrupt payloads throw a typed `TokzipDecodeError`
  (fuzz-tested: mutated, truncated, and garbage payloads either throw it or decode to the
  exact original — never silently wrong output).

The format is **pre-release (v2) and still evolving**: a version bump invalidates previously
written payloads, and decoders reject other versions instead of misdecoding them.

The wire format is specified in [FORMAT.md](FORMAT.md); the design rationale lives in
[issue #2](https://github.com/WillBooster/tokzip/issues/2).

## Streaming

`TokzipCompressionStream` / `TokzipDecompressionStream` are Web Streams
(`TransformStream<Uint8Array | string, Uint8Array>`), so the same code pipes in Node.js 18+
and browsers — mirroring the built-in `CompressionStream` API. The whole mechanism is hidden
inside the stream object: input is cut into blocks (256 KB by default), the LZ window is
carried across block boundaries, and every block independently ships the smaller of a
stored or range-coded body.

```ts
import { TokzipCompressionStream, TokzipDecompressionStream } from './src/index.ts';
import './src/languages/typescript.ts';

const compressed = readable.pipeThrough(new TokzipCompressionStream({ language: 'typescript' }));
const restored = compressed.pipeThrough(new TokzipDecompressionStream());
```

Streams use their own block container (binary channel only) and stay close to one-shot
ratios — on multi-megabyte inputs they even bench a few percent smaller than one-shot
frames, whose input exceeds the optimal parser's bound while stream blocks stay inside it.
Memory stays O(blockSize + window) on both sides regardless of stream length. Options: the
256 KB `blockSize` default is the practical ceiling (larger blocks shrink the history
budget and past 512 KB lose the optimal parse), `carryWindow: false` makes blocks
independently decodable, and `historyLimit` bounds the carried window (compression-speed
lever for small blocks); run
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

The benchmark is built around the intended deployment: **documents are compressed into a
database and dictionaries ship with the application**. That shapes two decisions:

- **The primary metric is the classic per-document ratio** — no per-session dictionary
  transfer applies when modules are bundled. Short documents (≤ 4 KB — the primary
  workload: code submissions and LLM outputs stored per-save) are additionally reported on
  their own.
- **The session-amortized, dictionary-inclusive ratio stays as a secondary metric** for
  session-delivered deployments: each language's bench docs are treated as one client
  session, tokzip is charged the brotli-compressed transfer size of that language's module
  once per session (competitors carry no dictionary), and each language gets a
  **breakeven analysis** — the cumulative input volume at which the dictionary pays for
  itself against browser-native `CompressionStream` gzip, the reference codec a client
  gets without shipping one (brotli/zstd/xz appear as server-side/CLI references).

The harness measures two channels separately. The **text channel** (default) compares
URL-safe text output: binary codecs use unpadded base64url, while tokzip and the `lz-string`
URI mode already emit text. The **binary channel** (`--binary`) compares tokzip binary
frames against the raw codec bytes with no text framing. Both channels **verify every
method on every document round-trips losslessly** — any mismatch fails the run — and
`--speed` additionally measures median end-to-end per-document throughput (CI runs the
speed pass on the text channel only).

Current numbers live on the dashboard (the v2 format reset — one range-coded mode — and
each dictionary retrain can change compressed output sizes, so older pinned tables no
longer apply). Two stable findings from the metric redesign: with the previous ~1 MB dictionaries
the brotli-compressed dictionary transfer (~300 KB per language) never paid for itself
against browser-native gzip on KB-scale sessions, and dictionary-free tokzip beats
`CompressionStream` gzip by roughly 2× on ≤ 1 KB documents. The default budget targets the
storage deployment — dictionaries ship with the application, so per-document
ratio and the first-compress index cost govern. Ratio improves monotonically with budget
(typescript ≤ 4 KB bench documents: 30.0% @16 KB vs 27.8% @128 KB, measured with the
current trainer), but the index build time and retained memory grow linearly with it, so 16 KB —
the smallest budget that still beats brotli -q11 in every text-channel bucket — is the
chosen balance; ratio-first deployments should retrain larger and session-delivered ones
smaller. Dictionaries are
trained exclusively on the public corpus — private production content never flows into
them.

The figures below are regenerated from the newest `main` benchmark run on every push, so
they can be newer than the pinned table above:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://willbooster.github.io/tokzip/charts/ratio-speed-dark.svg" />
  <img alt="Scatter chart of compression speed versus output size for tokzip and the baseline codecs on the newest main run" src="https://willbooster.github.io/tokzip/charts/ratio-speed-light.svg" />
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://willbooster.github.io/tokzip/charts/languages-dark.svg" />
  <img alt="Dot plot of per-language compression ratios for tokzip, CompressionStream gzip, and brotli q11 on the newest main run" src="https://willbooster.github.io/tokzip/charts/languages-light.svg" />
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
