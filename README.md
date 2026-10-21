# tokzip

[![npm version](https://img.shields.io/npm/v/tokzip.svg)](https://www.npmjs.com/package/tokzip)
[![license](https://img.shields.io/npm/l/tokzip.svg)](https://www.npmjs.com/package/tokzip)
[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![Test rust](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
[![wbfy](https://img.shields.io/badge/wbfy-20.43.1-1e90ff.svg)](https://github.com/WillBooster/shared/tree/main/packages/wbfy)

Lossless compressor for **prompts, LLM outputs, and source code** stored at rest — one
function in, one function out, no compression options (only an optional decode length limit).
The codec is Rust compiled to a single **wasm module (~1.1 MB, with 21 dictionaries
and trained models)** with a thin TypeScript wrapper; it runs on Node, Bun,
and Cloudflare Workers.
The module requires WebAssembly SIMD, which [Workers supports](https://developers.cloudflare.com/workers/runtime-apis/webassembly/#simd).

```ts
import { compress, decompress } from 'tokzip';

const frame = compress(text); // Uint8Array; strings must be well-formed UTF-16 (no lone surrogates)
const restored = decompress(frame); // === text
decompress(untrustedFrame, { maxLength: 8 * 1024 * 1024 }); // throws instead of expanding further
```

- **Node and Bun** load the module from the `wasm/` file shipped in the package (bundlers such
  as Next.js trace the `new URL(..., import.meta.url)` reference and carry the file along).
- **Cloudflare Workers** resolve the `workerd` export condition (wrangler and the Cloudflare
  Vite plugin set it) to an entry that imports the `.wasm` as a module, which is the only way
  Workers accept wasm. The wrapper imports `node:buffer`, so the Worker needs the
  `nodejs_compat` compatibility flag. The first call in an isolate decodes the dictionary of
  each language it uses and initializes shared model priors lazily; later calls reuse them.
  First-use CPU depends on the languages and the runtime, so measure it separately from warm calls.
- `FORMAT_VERSION` is 2. Future package releases will retain decoding support for format 2,
  including its dictionaries and trained model priors. A future format can improve compression
  for new writes without requiring existing records to be recompressed. Earlier experimental
  formats are not supported. `TokzipDecodeError` carries a numeric `code`. See [FORMAT.md](FORMAT.md)
  for the wire specification and compatibility rules.

There is nothing to configure: the encoder **detects the language of the input itself,
segment by segment** — a Japanese prompt, a Markdown answer with an HTML block whose
`<script>` is JavaScript, a TypeScript file — and codes each segment with the matching
trained dictionary and model priors. Embedded languages: `text`, `en-US`, `ja-JP`, `zh-CN`,
`zh-TW`, `html`, `css`, `javascript`, `typescript`, `c`, `cpp`, `csharp`, `dart`,
`haskell`, `java`, `jsp`, `php`, `python`, `ruby`, `rust`, `zig`; anything else falls back to
the closest one.

- **Codec**: LZ77 over the document plus the segment's dictionary (128 KB for prose
  languages; for code, a 64 KB part shared by every programming language plus 48 KB per
  language; COVER-trained: the fragments of the public corpus whose 8-byte grams occur in the
  most documents — counted over the public corpus and, for the languages it holds at least
  1 MiB of, the private production corpus, so every dictionary byte is a substring of a public document
  while the selection follows production usage; segment size picked by coding held-out
  documents), price-based parse (a bounded shortest-path over 1 KB chunks, committing early
  on long matches), adaptive binary range coder
  (LZMA-style symbol layout, terminated on the shortest representation of its final interval)
  whose literals are modeled by an order-2 context (128 trained classes of the previous byte
  × 32 of the one before) and whose models start from trained priors — short documents get
  the benefit of the statistics immediately, long documents adapt to themselves. The literal
  priors are shared in memory by the languages of a group (Latin prose, Japanese, Chinese, code) and
  trained on the group's pooled statistics — public and private documents alike, since priors
  are statistics, not content; the rest is per language. Matches into the
  dictionary are coded as absolute dictionary offsets, so the same fragment costs the same
  wherever it is referenced.
- **Detection without a parser**: one table maps every 4-gram hash to the languages whose
  dictionary contains it; the input is scored per 64-byte window, sampling alternate positions
  on documents of at least 128 bytes. Labeled code fences add a hint. A Viterbi pass with a
  switch penalty turns the scores into segments, whose
  boundaries snap to the nearest line start. The table stores 16-bit indices into a palette
  of language sets; dense sets are scored by subtracting misses instead of adding hits. On mixed
  prose + code documents this codes ~4 pp smaller than the best single language. When the
  split is uncertain (several segments, or a single language that is not the strongest
  dictionary match) the encoder also codes the document as one segment of the strongest
  match and keeps the smaller — comparing the two on the first 2 KB only above 4 KB, so
  the amount of trial input stays bounded. Identical trial segmentations are encoded only once
  as the selected full document.
- **Storage-grade**: every frame carries a CRC-32 of its content, `compress` decodes
  every coded block it builds and compares the result to the input before accepting it
  (storing the content — or, in a multi-block frame, that block — verbatim otherwise), and the
  decoder checks framing and CRC-32, reporting failed checks with a typed `TokzipDecodeError`. Incompressible input never expands beyond the stored-frame header (5 bytes
  plus the length varint). Content above 4 MiB is coded as independent 4 MiB blocks (a block
  whose first 256 KiB does not shrink is stored without coding the rest), so the coder's
  working set (several copies of a block) stays bounded whatever the document size; only the
  input and the output (held twice while the frame is assembled) scale with it. The format
  bounds a frame's expansion only relative to its size (a small frame of repetitive content
  legitimately expands thousands of times), so pass `maxLength` when decompressing frames from
  an untrusted source.

## Results

Bench split of the pinned [`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus)
plus the private production corpus when checked out beside it (21 languages),
compressed size as a percentage of the input:

| documents  | tokzip    | brotli -11 | zstd -19 | gzip -9 |
| ---------- | --------- | ---------- | -------- | ------- |
| all        | **20.8%** | 26.3%      | 31.1%    | 31.9%   |
| ≤ 1 KB     | **28.5%** | 47.6%      | 60.2%    | 61.1%   |
| 1–4 KB     | **23.0%** | 33.9%      | 42.1%    | 42.2%   |
| 4–16 KB    | **20.8%** | 25.8%      | 30.4%    | 31.2%   |
| > 16 KB    | **19.1%** | 21.6%      | 24.5%    | 25.6%   |
| ja-JP      | **24.9%** | 35.6%      | 42.8%    | 43.3%   |
| zh-CN      | **26.4%** | 36.8%      | 45.3%    | 46.1%   |
| typescript | **14.9%** | 17.7%      | 19.8%    | 20.6%   |
| java       | **13.9%** | 23.3%      | 28.2%    | 28.6%   |
| html       | **17.8%** | 21.3%      | 25.4%    | 26.1%   |

Cloudflare Workers allow [128 MB per isolate and 10 ms of CPU per request on the free plan](https://developers.cloudflare.com/workers/platform/limits/).
Dictionary initialization and larger documents can exceed the free CPU budget; local throughput
is not a reliable way to predict production Workers CPU.

Run it yourself: `bun run bench` (add `--speed` for throughput and `--json <file>` for a
machine-readable report). `--codec-only --repeat 3` measures tokzip alone after warming the
corpus. With `--speed`, JSON reports include throughput and p50/p95/p99/max latency for the whole
corpus and each language, size band, and corpus. Reports include the runtime, corpus
fingerprint, and Wasm hash; every measured frame is checked for lossless round-trip recovery.
To measure serialized application responses, pass `--input <documents.jsonl>`: each non-blank
line is one complete JSON value, compressed as written, without the line separator. This uses
only the supplied file. Reports contain aggregate measurements and fingerprints, not the input
records.

## License

Original tokzip code is licensed under [Apache-2.0](LICENSE). Embedded dictionary fragments
and compiled dependencies retain their upstream terms. Redistributors must retain the
accompanying [NOTICE](NOTICE) and [third-party license material](THIRD_PARTY_NOTICES/index.json).

## Development

Read the [dependency patch notes](https://github.com/WillBooster/tokzip/blob/main/patches/README.md)
before updating the patched development dependencies.

```bash
mise install          # bun, node, rust (+ wasm32-unknown-unknown)
bun install
bun run build         # rust → wasm/tokzip.wasm (committed; rebuild after codec changes), then builds dist/
bun test              # round-trip and resilience tests through the wasm build
cargo test --release --manifest-path rust/Cargo.toml
bun run train         # new-format development only: read FORMAT.md before overwriting frozen assets
```

Layout: `rust/crates/tokzip` (codec: `lz.rs` token coding + `lz/parse.rs` price-based search,
`lang.rs` dictionaries + detection, `languages.rs` the language table, `train.rs` dictionary + priors trainer,
`pack.rs` + `build.rs` asset packing), `rust/crates/tokzip-wasm` (C-ABI exports), `src/`
(wrapper: `core.ts` over a compiled module, `index.ts` for Node and Bun, `workers.ts` for
Cloudflare Workers), `dict/` (the wrapper, the code group's shared part, and each language's
suffix) and `priors/` (each group's packed literal priors and each language's own model
nodes; the build embeds them as committed, each dictionary part coded by the codec itself,
and the detection table of every dictionary's 4-grams), `scripts/train` (wrapper dictionary + trainer entry), `scripts/bench` (corpus
benchmark), `bench/cloudflare` (Workers benchmark: `bunx wrangler deploy --config
bench/cloudflare/wrangler.jsonc`, then `node bench/cloudflare/measure.ts <worker-url>`). The
Workers driver records request CPU from `wrangler tail` and client time-to-first-byte for
prose, code, JSON responses, and a large mixed document. The driver discards requests on an
isolate's first invocation. `fresh-*` rows create a fresh
Wasm instance and initialize its dictionaries; other codec rows additionally require the
sample to be prepared. Decoder frames are prepared locally, so a fresh decode never needs a
compression in the measured request.

Codec iteration aids (`--features train`): `cargo run --release --features train --example
eval -- <corpus>:<language> ...` reports ratio and speed per language (`TOKZIP_COST=1` adds
the share of output bits per model group), `--example prof` the per-step cost of small
documents.
