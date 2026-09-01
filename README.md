# tokzip

[![Test rust](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test-rust.yml)
[![Test](https://github.com/WillBooster/tokzip/actions/workflows/test.yml/badge.svg)](https://github.com/WillBooster/tokzip/actions/workflows/test.yml)
[![wbfy](https://img.shields.io/badge/wbfy-19.2.1-1e90ff.svg)](https://github.com/WillBooster/shared/tree/main/packages/wbfy)

Lossless compressor for **prompts, LLM outputs, and source code** stored at rest — one
function in, one function out, no options. The codec is Rust compiled to a single
**wasm module (~450 KB, every dictionary included)** with a thin TypeScript wrapper; it runs
on Cloudflare Workers and Bun (and on Node via a bundler that handles `.wasm` imports).

```ts
import { compress, decompress } from 'tokzip';

const frame = compress(text); // Uint8Array; strings must be well-formed UTF-16 (no lone surrogates)
const restored = decompress(frame); // === text (string in → string out, bytes in → bytes out)
```

There is nothing to configure: the encoder **detects the language of the input itself,
segment by segment** — a Japanese prompt, a Markdown answer with an HTML block whose
`<script>` is JavaScript, a TypeScript file — and codes each segment with the matching
trained dictionary and model priors. Embedded languages: `text`, `en-US`, `ja-JP`, `html`,
`css`, `javascript`, `typescript`; anything else falls back to the closest one.

- **Codec**: LZ77 over the document plus the segment's dictionary, price-based parse (a
  bounded shortest-path over 4 KB chunks, not a global optimum),
  adaptive binary range coder (LZMA-style symbol layout) whose models start from trained
  per-language priors — short documents get the benefit of the statistics immediately, long
  documents adapt to themselves. Matches into the dictionary are coded as absolute
  dictionary offsets, so the same fragment costs the same wherever it is referenced.
- **Detection without a parser**: every language keeps a bitset of the 4-grams of its
  dictionary; the input is scored per 64-byte window (labeled code fences add a hint), and a
  Viterbi pass with a switch penalty turns the scores into segments. Cost: a few table
  lookups per byte. On mixed prose + code documents this codes ~2 pp smaller than the best
  single language.
- **Storage-grade**: every frame carries a CRC-32 of its content and type, `compress` decodes its own
  output and compares it to the input before returning (falling back to a stored frame
  otherwise), and a corrupt or truncated frame either throws a typed `TokzipDecodeError` or
  (when the range coder's trailing slack absorbs the damage) decodes to the exact original —
  never silently wrong output. Incompressible input never expands beyond the stored-frame
  header (7–10 bytes, depending on the length varint). Content above 4 MiB is stored verbatim
  rather than coded (coding holds several copies of the input in memory), so chunk larger
  payloads before compressing them.
- **Format v0 (pre-release)**: the format changes freely with no compatibility for earlier
  frames until it is fixed as v1; decoders reject other versions. See [FORMAT.md](FORMAT.md).

## Results

Bench split of the pinned [`tokzip-corpus`](https://github.com/WillBooster/tokzip-corpus)
(3,288 documents, 21 languages — 14 of them have no embedded dictionary), compressed size as
a percentage of the input:

| documents  | tokzip    | brotli -11 | zstd -19 | gzip -9 |
| ---------- | --------- | ---------- | -------- | ------- |
| all        | 26.9%     | 27.1%      | 32.2%    | 32.9%   |
| ≤ 1 KB     | **41.8%** | 47.8%      | 60.3%    | 61.2%   |
| ≤ 4 KB     | **33.0%** | 34.0%      | 42.3%    | 42.3%   |
| ja-JP      | **31.9%** | 35.6%      | 42.8%    | 43.3%   |
| typescript | **15.6%** | 17.1%      | 19.1%    | 19.9%   |
| html       | **18.8%** | 21.5%      | 27.3%    | 27.6%   |

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
detection, `train.rs` priors trainer), `rust/crates/tokzip-wasm` (C-ABI exports),
`src/index.ts` (wrapper), `dict/` and `priors/` (trained assets embedded into the wasm),
`scripts/train` (dictionary trainer), `scripts/bench` (corpus benchmark),
`bench/cloudflare` (Workers benchmark; needs a `wrangler` on `PATH`, which this repository does
not declare: `wrangler deploy --config bench/cloudflare/wrangler.jsonc`, then
`bun bench/cloudflare/measure.ts`).
