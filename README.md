# tokzip

[![Test rust](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml)
[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![wbfy](https://img.shields.io/badge/wbfy-19.2.1-1e90ff.svg)](https://github.com/WillBooster/shared/tree/main/packages/wbfy)

Lossless compressor for **prompts, LLM outputs, and source code** stored at rest — one
function in, one function out, no compression options (only an optional decode length limit). The codec is Rust compiled to a single
**wasm module (~1.2 MB, every dictionary and model included; ~540 KB gzipped)** with a thin
TypeScript wrapper; it runs on Cloudflare Workers and Bun (and on Node via a bundler that
handles `.wasm` imports).

```ts
import { compress, decompress } from 'tokzip';

const frame = compress(text); // Uint8Array; strings must be well-formed UTF-16 (no lone surrogates)
const restored = decompress(frame); // === text
decompress(untrustedFrame, { maxLength: 8 * 1024 * 1024 }); // throws instead of expanding further
```

There is nothing to configure: the encoder **detects the language of the input itself,
segment by segment** — a Japanese prompt, a Markdown answer with an HTML block whose
`<script>` is JavaScript, a TypeScript file — and codes each segment with the matching
trained dictionary and model priors. Embedded languages: `text`, `en-US`, `ja-JP`, `html`,
`css`, `javascript`, `typescript`; anything else falls back to the closest one.

- **Codec**: LZ77 over the document plus the segment's 128 KB dictionary (COVER-trained:
  the fragments whose 8-byte grams recur most across the corpus, segment size picked by coding
  held-out documents), price-based parse (a bounded shortest-path over 4 KB chunks, not a
  global optimum), adaptive binary range coder (LZMA-style symbol layout) whose literals are
  modeled by an order-2 context (128 trained classes of the previous byte × 4 of the one
  before) and whose models start from trained per-language priors — short documents get the
  benefit of the statistics immediately, long documents adapt to themselves. Matches into the
  dictionary are coded as absolute dictionary offsets, so the same fragment costs the same
  wherever it is referenced.
- **Detection without a parser**: every language keeps a bitset of the 4-grams of its
  dictionary; the input is scored per 64-byte window (labeled code fences add a hint), and a
  Viterbi pass with a switch penalty turns the scores into segments, whose boundaries snap to
  the nearest line start. Cost: a few table lookups per byte. On mixed prose + code documents
  this codes ~4 pp smaller than the best single language.
- **Storage-grade**: every frame carries a CRC-32 of its content, `compress` decodes
  every coded block it builds and compares the result to the input before accepting it
  (storing the content — or, in a multi-block frame, that block — verbatim otherwise), and a
  corrupt or truncated frame either throws a typed `TokzipDecodeError` or (when the range
  coder's trailing slack absorbs the damage) decodes to the exact original — never silently
  wrong output. Incompressible input never expands beyond the stored-frame header (5 bytes
  plus the length varint). Content above 4 MiB is coded as independent 4 MiB blocks (a block
  whose first 256 KiB does not shrink is stored without coding the rest), so the coder's
  working set (several copies of a block) stays bounded whatever the document size; only the
  input and the output (held twice while the frame is assembled) scale with it. The format bounds a frame's expansion only relative to its
  size (a small frame of repetitive content legitimately expands thousands of times), so pass
  `maxLength` when decompressing frames from an untrusted source.
- **Format v0 (pre-release)**: the format changes freely with no compatibility for earlier
  frames until it is fixed as v1; decoders reject other versions. The dictionaries and priors
  are part of the format, so from v1 on every retrain is a new version whose decoder keeps the
  earlier generations readable while `compress` writes only the newest. See
  [FORMAT.md](FORMAT.md).

## Results

Bench split of the pinned [`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus)
plus the private production corpus when checked out beside it (3,514 documents, 21 languages —
14 of them have no embedded dictionary), compressed size as a percentage of the input:

| documents  | tokzip    | brotli -11 | zstd -19 | gzip -9 |
| ---------- | --------- | ---------- | -------- | ------- |
| all        | **23.3%** | 26.3%      | 31.1%    | 31.9%   |
| ≤ 1 KB     | **33.6%** | 47.6%      | 60.2%    | 61.1%   |
| 1–4 KB     | **27.8%** | 33.9%      | 42.1%    | 42.2%   |
| 4–16 KB    | **23.4%** | 25.8%      | 30.4%    | 31.2%   |
| > 16 KB    | **20.3%** | 21.6%      | 24.5%    | 25.6%   |
| ja-JP      | **25.1%** | 35.6%      | 42.8%    | 43.3%   |
| typescript | **14.8%** | 17.7%      | 19.8%    | 20.6%   |
| html       | **19.3%** | 21.3%      | 25.4%    | 26.1%   |

Run it yourself: `bun run bench` (add `--speed` for throughput and `--json <file>` for a
machine-readable report). The harness verifies every tokzip frame round-trips losslessly.

## Development

```bash
mise install          # bun, node, rust (+ wasm32-unknown-unknown)
bun install
bun run build         # rust → wasm/tokzip.wasm (committed; rebuild after codec changes)
bun test              # round-trip and resilience tests through the wasm build
cargo test --release --manifest-path rust/Cargo.toml
bun run train         # retrain dict/*.bin and priors/*.bin from ../tokzip-corpus (then build)
```

Layout: `rust/crates/tokzip` (codec: `lz.rs` parse + coder, `lang.rs` dictionaries +
detection, `train.rs` dictionary + priors trainer), `rust/crates/tokzip-wasm` (C-ABI
exports), `src/index.ts` (wrapper), `dict/` and `priors/` (trained assets embedded into the
wasm; `build.rs` packs the priors, skipping their untrained literal subtrees),
`scripts/train` (wrapper dictionary + trainer entry), `scripts/bench` (corpus benchmark),
`bench/cloudflare` (Workers benchmark; needs a `wrangler` on `PATH`, which this repository does
not declare: `wrangler deploy --config bench/cloudflare/wrangler.jsonc`, then
`bun bench/cloudflare/measure.ts`).
