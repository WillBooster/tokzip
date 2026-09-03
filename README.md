# tokzip

[![Test rust](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml)
[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![wbfy](https://img.shields.io/badge/wbfy-19.2.1-1e90ff.svg)](https://github.com/WillBooster/shared/tree/main/packages/wbfy)

Lossless compressor for **prompts, LLM outputs, and source code** stored at rest — one
function in, one function out, no compression options (only an optional decode length limit).
The codec is Rust compiled to a single **wasm module (~1.2 MB, all 21 dictionaries and models
included)** with a thin TypeScript wrapper; it runs on Node, Bun, and Cloudflare Workers.

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
  each language it uses (~10 ms of CPU time for the first `compress`, ~7 ms for the first
  `decompress`, ~5 ms per further language); later calls do not.
- `FORMAT_VERSION` is the version `compress` writes (store it beside frames if you ever want
  to know which codec generation produced them); `TokzipDecodeError` carries a numeric `code`.

There is nothing to configure: the encoder **detects the language of the input itself,
segment by segment** — a Japanese prompt, a Markdown answer with an HTML block whose
`<script>` is JavaScript, a TypeScript file — and codes each segment with the matching
trained dictionary and model priors. Embedded languages: `text`, `en-US`, `ja-JP`, `zh-CN`,
`zh-TW`, `html`, `css`, `javascript`, `typescript`, `c`, `cpp`, `csharp`, `dart`,
`haskell`, `java`, `jsp`, `php`, `python`, `ruby`, `rust`, `zig`; anything else falls back to
the closest one.

- **Codec**: LZ77 over the document plus the segment's dictionary (128 KB for prose
  languages; for code, a 64 KB part shared by every programming language plus 48 KB per
  language; COVER-trained: the fragments whose 8-byte grams occur in the most documents of
  the corpus, segment size picked by coding held-out documents), price-based parse (a bounded
  shortest-path over 4 KB chunks, not a global optimum), adaptive binary range coder
  (LZMA-style symbol layout, terminated on the shortest representation of its final interval)
  whose literals are modeled by an order-2 context (128 trained classes of the previous byte
  × 32 of the one before) and whose models start from trained priors — short documents get
  the benefit of the statistics immediately, long documents adapt to themselves. The literal
  priors are shared by the languages of a group (Latin prose, Japanese, Chinese, code) and
  trained on the group's pooled statistics; the rest is per language. Matches into the
  dictionary are coded as absolute dictionary offsets, so the same fragment costs the same
  wherever it is referenced.
- **Detection without a parser**: one table maps every 4-gram hash to the languages whose
  dictionary contains it; the input is scored per 64-byte window (labeled code fences add a
  hint), and a Viterbi pass with a switch penalty turns the scores into segments, whose
  boundaries snap to the nearest line start. Cost: one table lookup per byte. On mixed
  prose + code documents this codes ~4 pp smaller than the best single language. When the
  split is uncertain (several segments, or a single language that is not the strongest
  dictionary match) the encoder also codes the document as one segment of the strongest
  match and keeps the smaller — comparing the two on the first 8 KB only above that size, so
  the second parse never costs more than a few milliseconds.
- **Storage-grade**: every frame carries a CRC-32 of its content, `compress` decodes
  every coded block it builds and compares the result to the input before accepting it
  (storing the content — or, in a multi-block frame, that block — verbatim otherwise), and a
  corrupt or truncated frame either throws a typed `TokzipDecodeError` or (when the range
  coder's trailing slack absorbs the damage) decodes to the exact original — never silently
  wrong output. Incompressible input never expands beyond the stored-frame header (5 bytes
  plus the length varint). Content above 4 MiB is coded as independent 4 MiB blocks (a block
  whose first 256 KiB does not shrink is stored without coding the rest), so the coder's
  working set (several copies of a block) stays bounded whatever the document size; only the
  input and the output (held twice while the frame is assembled) scale with it. The format
  bounds a frame's expansion only relative to its size (a small frame of repetitive content
  legitimately expands thousands of times), so pass `maxLength` when decompressing frames from
  an untrusted source.
- **Format v1**: the format is the codec — algorithm, dictionaries, and priors together. Any
  change to any of them (a retrained dictionary as much as a new model) is a new version;
  `compress` writes only the newest and `decompress` keeps reading every earlier one, so
  versions are rare and deliberate. See [FORMAT.md](FORMAT.md).

## Results

Bench split of the pinned [`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus)
plus the private production corpus when checked out beside it (3,514 documents, 21 languages),
compressed size as a percentage of the input:

| documents  | tokzip    | brotli -11 | zstd -19 | gzip -9 |
| ---------- | --------- | ---------- | -------- | ------- |
| all        | **21.0%** | 26.3%      | 31.1%    | 31.9%   |
| ≤ 1 KB     | **28.6%** | 47.6%      | 60.2%    | 61.1%   |
| 1–4 KB     | **23.1%** | 33.9%      | 42.1%    | 42.2%   |
| 4–16 KB    | **21.0%** | 25.8%      | 30.4%    | 31.2%   |
| > 16 KB    | **19.4%** | 21.6%      | 24.5%    | 25.6%   |
| ja-JP      | **25.0%** | 35.6%      | 42.8%    | 43.3%   |
| zh-CN      | **26.6%** | 36.8%      | 45.3%    | 46.1%   |
| typescript | **15.0%** | 17.7%      | 19.8%    | 20.6%   |
| java       | **14.0%** | 23.3%      | 28.2%    | 28.6%   |
| html       | **19.3%** | 21.3%      | 25.4%    | 26.1%   |

Throughput through the wasm build in Bun on an Apple M-series laptop: compression ~2.8 MB/s
(the price-based parse is the cost; a 4 KB document takes ~1.3 ms), decompression ~90 MB/s.
Cloudflare's free plan allows 10 ms of CPU per request, so a document above ~20 KB (or the
first `compress` in an isolate, ~10 ms) needs a paid plan or a place outside the request path.

Run it yourself: `bun run bench` (add `--speed` for throughput and `--json <file>` for a
machine-readable report). The harness verifies every tokzip frame round-trips losslessly.

## Development

```bash
mise install          # bun, node, rust (+ wasm32-unknown-unknown)
bun install
bun run build         # rust → wasm/tokzip.wasm (committed; rebuild after codec changes), then builds dist/
bun test              # round-trip and resilience tests through the wasm build
cargo test --release --manifest-path rust/Cargo.toml
bun run train         # retrain dict/*.bin and priors/*.bin from ../tokzip-corpus (then build)
```

Layout: `rust/crates/tokzip` (codec: `lz.rs` parse + coder, `lang.rs` dictionaries +
detection, `languages.rs` the language table, `train.rs` dictionary + priors trainer,
`pack.rs` + `build.rs` asset packing), `rust/crates/tokzip-wasm` (C-ABI exports), `src/`
(wrapper: `core.ts` over a compiled module, `index.ts` for Node and Bun, `workers.ts` for
Cloudflare Workers), `dict/` (the wrapper, the code group's shared part, and each language's
suffix) and `priors/` (each group's packed literal priors and each language's own model
nodes; the build embeds them as committed, each dictionary part coded by the codec itself,
and the detection table of every dictionary's 4-grams), `scripts/train` (wrapper dictionary + trainer entry), `scripts/bench` (corpus
benchmark), `bench/cloudflare` (Workers benchmark; needs a `wrangler` on `PATH`, which this
repository does not declare: `wrangler deploy --config bench/cloudflare/wrangler.jsonc`, then
`bun bench/cloudflare/measure.ts`).

Codec iteration aids (`--features train`): `cargo run --release --features train --example
eval -- <corpus>:<language> ...` reports ratio and speed per language (`TOKZIP_COST=1` adds
the share of output bits per model group), `--example prof` the per-step cost of small
documents.
