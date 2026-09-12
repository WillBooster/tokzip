# Codec optimization measurements — 2026-09-09–10 (JST)

This experiment starts at `main` commit `5701b10`. It follows the private-corpus training
work in [#45](https://github.com/WillBooster/tokzip/pull/45), the bounded alternative parse and
shared code dictionary in [#44](https://github.com/WillBooster/tokzip/pull/44), and the
precomputed detection table in [#43](https://github.com/WillBooster/tokzip/pull/43).
Measurements and exploratory results are in [optimization.json](optimization.json).
The main tables describe the SIMD milestone (`assets.simd` in the JSON). The subsequent
literal-cache revision and current module (`assets.final`) are recorded separately below.

## Data and method

- Public corpus: `a62302c7e9481b80b756dee1a152ace5858f65de`; private corpus:
  `71983c19077854cb86d35eddcd34794047a7152e`. The standard benchmark later pulled private
  commit `7cab86690cf2b401698468bfae08e717aa32a12f`; its corpus tree is identical. Both tree
  hashes are recorded in the JSON.
- Bench split: 3,514 documents, 16,264,597 UTF-8 bytes, 21 languages. The private portion
  has 226 documents / 2,441,295 bytes, all identified as ai-game-builder samples. This does
  not establish performance on fresh Exercode or qa-support production traffic.
- Training reads training documents; dictionary bytes come only from public documents.
  Private documents influence fragment selection and aggregate priors. Candidate selection
  uses the bench split, so this is an optimization set, not an untouched final holdout.
- Local host: Apple M3, macOS arm64; Node 24.20.0, Bun 1.4.2, Rust 1.98.0. Release profile:
  optimization level 3, LTO, one codegen unit, stripped Wasm. No new runtime dependencies.
- Local A/B: load both compiled modules and warm the corpus, then six full paired passes,
  reversing baseline/candidate order each pass. Report the last four passes; the first two
  are stabilization passes. Every compressed frame is decoded and compared with the input.
  Throughput uses UTF-8 input bytes; compression includes the codec's own recovery check.
- Ratios are total frame bytes / total input bytes, not averages of document percentages.
  All sizes below include framing. Smaller ratios mean better compression.

## Retained implementation

1. Visit each probability-tree node once to build leaf prices and cache token-flag prices
   once per parse chunk. Replace the coarse 128-entry price table with 2,048 one-byte prices.
   An integer fixed-point calculation generates the table at build time, with enough
   precision to match rounded `-16 * log2(p / 2048)` for every valid probability. The parse
   still uses bounded 4 KiB chunks, so it is not a globally optimal parse.
2. Store detection masks as 16-bit palette indices plus three-byte language sets. For dense
   masks, subtract misses instead of adding hits: subtracting a common score from every
   language preserves all score differences, Viterbi choices, and tie ordering.
3. Allocate immutable literal priors and their context tables once per language group.
   Per-document adaptation still uses independent, pooled copy-on-first-use arenas; it never
   mutates the shared priors. Keep the context-table reference directly in the document model.
4. Encode embedded dictionary suffixes against their already available wrapper/group prefix.
   This changes asset packing, not the dictionary bytes that a decoder reconstructs.
5. Skip duplicate alternative-parse trials when their prefix segmentations are identical;
   the existing tie rule selects the detected segmentation. Decode the Wasm output view
   directly into a string, avoiding an intermediate JavaScript byte copy.
6. Retrain dictionaries and priors with the final price calculation. Retain the original
   match-length grammar and search heuristics. Mark generated dictionary/prior files binary
   so Git does not normalize their byte content or render them as text diffs.
7. Compare 16 bytes at a time in the Wasm encoder's common-prefix scan with SIMD; native Rust
   and short tails use the scalar scan. Every vector load is bounded by both slice lengths,
   and [`v128_load` permits unaligned pointers](https://doc.rust-lang.org/core/arch/wasm32/fn.v128_load.html).
   The SIMD milestone added 43 bytes over its scalar build and produced identical frames on all 3,514 documents.

## Results

| Metric                                               |     Baseline | SIMD milestone |
| ---------------------------------------------------- | -----------: | -------------: |
| All documents, frame/input                           |     20.8636% |       20.8251% |
| Public documents                                     |     21.2562% |       21.2204% |
| Private documents                                    |     18.6407% |       18.5867% |
| Documents ≤ 1 KiB                                    |     28.6927% |       28.7018% |
| Wasm bytes                                           |    1,243,552 |      1,148,077 |
| Wasm gzip -9 bytes                                   |      938,467 |        956,226 |
| Node compression, median MB/s                        |        2.974 |          3.476 |
| Node decompression, median MB/s                      |        94.53 |          91.58 |
| Bun compression, median MB/s                         |        2.940 |          3.494 |
| Bun decompression, median MB/s                       |        99.01 |          97.19 |
| Wasm linear memory after 21 representative documents | 61,734,912 B |   34,996,224 B |
| Wasm linear memory after compressing the full corpus | 73,400,320 B |   42,991,616 B |
| Cumulative first-use elapsed time, Node median       |     61.01 ms |       50.87 ms |

Compression throughput improves about 17% in Node and 19% in Bun. Compressed bytes decrease
0.185% overall; the ratio improvement is modest, and the ≤1 KiB bucket slightly regresses.
Decompression throughput drops about 2–3% in this paired measurement. The retained tradeoff
reduces Wasm linear memory by 43.3% and the uncompressed module by 7.7%, while gzip packaging
grows 1.9%. The memory measurement uses a fresh instance and one 500–4,096-byte document per
language; it is not peak isolate memory for arbitrary large or mixed documents. The
first-use measurement excludes module compilation and is not a Workers cold-start estimate.
A separate pass over all 3,514 documents through the raw Wasm compression ABI uses 41.4% less
linear memory. Both measurements exclude JavaScript heap and compiled-code memory.

## Cloudflare Workers

The SIMD milestone comparison imports baseline and SIMD modules into one temporary Worker and posts
identical batches of up to 128 documents in alternating order. Each request returns an
isolate ID and whether that codec has already processed that entire batch. Unprepared calls
are discarded and repeated. A baseline/candidate pair is accepted only when both are prepared
and their isolate IDs match. Two isolates served the run; all 160 accepted pairs matched
within the pair. After one stabilization pass, three full passes are measured.

`wrangler tail` supplies CPU time for the exact request URL. There were 33 missing-CPU retries,
123 preparation retries, and one mismatched-isolate pair retry. Missing results are never
treated as zero. Every accepted request compresses, decompresses, and compares every document.
CPU includes JSON parsing, UTF-8 counting, compression, decompression, recovery comparison,
and response construction; it does not separate encoder and decoder time. Compatibility date
was 2026-09-09 with `nodejs_compat`, Wrangler 4.128.0, colo NRT.

| Full-corpus pass | Baseline CPU | SIMD CPU | Reduction |
| ---------------- | -----------: | -------: | --------: |
| 1                |     13.400 s | 11.461 s |     14.5% |
| 2                |     15.523 s | 12.436 s |     19.9% |
| 3                |     13.487 s | 11.289 s |     16.3% |
| Total            |     42.410 s | 35.186 s |     17.0% |

Across the three passes, public-corpus CPU decreases 17.6% and private-corpus CPU 14.7%.
This is a measured workload result, not a guarantee for every document or runtime instance.
Reports contain counts, sizes, CPU, and aggregate isolate information; corpus contents are
not embedded in deployed assets or committed reports.

The earlier scalar milestone was also tested in separate Workers and a shared Worker without
isolate tracking. Separate deployments reported 24.883 s (scalar milestone) versus 14.219 s (baseline)
median CPU, whereas the shared deployment reported 17.971 s versus 20.772 s in that order. Those uncontrolled results motivated
the prepared-batch, matching-isolate protocol above; they remain in the JSON as diagnostic
measurements, not evidence of a final speedup. Local `performance.now()` is not used to infer
production CPU. All four temporary Workers were deleted after measurement.

As of this measurement, Workers allow [128 MB per isolate, 64 MiB uncompressed Worker size,
and 10 ms CPU per free-plan request](https://developers.cloudflare.com/workers/platform/limits/);
there is no compressed Worker-size limit. Paid requests default to 30 seconds of CPU and can
be configured up to five minutes. The memory budget includes both JavaScript and Wasm, so
sharing primed models is useful even though the module itself is well below the size limit.

## Rejected experiments

These are exploratory, usually single-pass Node measurements at different intermediate
revisions, not isolated effects that can be added together. The JSON records the individual
results. The A/B results above describe the SIMD milestone; the cache revision is measured separately below.

| Experiment                                  | All ratio | Compression MB/s | Decision                                                          |
| ------------------------------------------- | --------: | ---------------: | ----------------------------------------------------------------- |
| Restart parsing at a long rep               |  20.8653% |            2.924 | Slower and larger                                                 |
| Separate dictionary/history match frontiers |  20.8602% |            2.930 | Small ratio gain, appreciable slowdown                            |
| Avoid speculative literal-tree copies       |  20.8636% |            3.112 | No stable speed benefit                                           |
| Force range-decoder inlining                |  20.8452% |            3.418 | Larger module and slower than the flag-price stage                |
| Match + literal + rep parse edge            |  20.8146% |            3.086 | Better ratio, excessive encoder cost                              |
| Separate HTML literal model                 |  20.8215% |            3.593 | Private ratio worsened to 18.6555%; module grew                   |
| Skip all interior positions of long matches |  20.8283% |            3.690 | Mixed-answer Worker CPU regressed in a same-deployment comparison |
| Extended 1,296-byte match plus skipping     |  20.8233% |            3.673 | Tiny ratio gain; no consistent mixed-answer Worker gain           |
| Order-1 prior shrinkage, weight 16          |  20.8313% |            3.718 | Smaller module, worse aggregate ratio than the retained model     |
| Order-1 prior shrinkage, weight 4           |  20.8260% |            3.547 | Negligible ratio gain for added training complexity               |

SIMD initially lost an early Node/Apple M3 screening (3.332 versus 3.413 MB/s). Retesting
on Workers with the final model/pricing changes showed a benefit, so it was retained and
revalidated on the full corpus in all three runtimes. The early result remains in the JSON.
Larger context models were already swept in [#44](https://github.com/WillBooster/tokzip/pull/44): 64 second-byte classes saved another
0.04 percentage points for 72 KB of priors. This iteration keeps 32 classes to preserve the
module-size reduction.

A final decoder-only experiment cached literal-probability and tree-index slices directly in
each document. It reduced Node throughput from about 84 to 70 MB/s on identical frames and
was discarded. Returning to the copying TypeScript decoder also lost in that comparison.
The additional references and fewer source-level operations did not translate into faster
Wasm execution.

## Literal-cache revision after review

The process-wide group cache holds only `Arc<LiteralModels>`. Language-specific probability
nodes are constructed for each language's `Models`; no unused language nodes are retained in
the group cache. The packed asset parser serves both the build script and runtime.

Against the SIMD milestone, this removes 13,424 bytes of unused probability values. The
module shrinks from 1,148,077 to 1,147,923 bytes (gzip -9: 956,226 to 956,130 bytes).
Full-corpus Wasm linear memory falls from 42,991,616 to 42,795,008 bytes (192 KiB); the
21-document first-use memory remains 34,996,224 bytes. All 3,514 frames are byte-identical
and cross-decode in both directions, so every compressed-size ratio above is unchanged.

Six paired local passes per run use the same JavaScript wrapper for both modules. Medians
of the last four passes are below; each measured frame is checked against its source.
The additional Bun run investigates the first run's compression slowdown.

| Runtime/run     | Before compression MB/s | After compression MB/s | Before decompression MB/s | After decompression MB/s |
| --------------- | ----------------------: | ---------------------: | ------------------------: | -----------------------: |
| Node            |                   3.503 |                  3.483 |                     92.37 |                    92.97 |
| Bun, first run  |                   3.878 |                  3.690 |                    100.63 |                   101.35 |
| Bun, repeat run |                   3.723 |                  3.690 |                    100.43 |                   101.41 |

The local runs show lower memory/size and slower compression: Node compression is
0.6% slower and Bun is 4.8% slower in the first run, 0.9% in the repeat run. Decompression
improves slightly. Five fresh-instance Node runs give median first-use elapsed times of
50.02 ms before and 51.20 ms after; these exclude compilation and are not Workers CPU.
`reviewCacheCleanup` in the JSON contains the paired measurements, fingerprints and memory
results. Rust release tests (13), native and Wasm Clippy, and `bun run verify-full` pass.

A separate Workers comparison uses the same prepared-batch, matching-isolate protocol as
above, comparing the SIMD milestone with the cache revision. Three measured full-corpus
passes, after a full warmup pass, give:

| Full-corpus pass | Before CPU | After CPU |
| ---------------- | ---------: | --------: |
| 1                |    8.827 s |   8.640 s |
| 2                |    8.733 s |   8.841 s |
| 3                |    8.409 s |   8.521 s |
| Total            |   25.969 s |  26.002 s |

The total CPU increase is 0.1%, with mixed directions per pass. The cache revision is retained
for its smaller module, lower memory footprint, and simpler ownership; no additional speedup
is claimed. These measurements compare against the SIMD milestone, independently of the
main-to-SIMD comparison above; their percentages must not be combined.
All 160 accepted pairs are prepared and share an isolate within each pair (two isolates,
NRT). There are 41 missing-CPU retries, 88 batch-preparation retries and one isolate mismatch.
The additional temporary Worker was deleted. Its data and deployment metadata are in
`reviewCacheCleanup.workersPaired`.

## Reproduction and compatibility

`bun run bench --speed --repeat 3 --json .tmp/bench.json` compares the current codec with
Brotli, Zstd, and gzip. Add `--codec-only` to omit those codecs. The report records runtime,
Wasm SHA-256, and a corpus fingerprint; pin both corpus checkouts to compare revisions.
Repeated runs exclude first-use initialization. The default single pass retains first-use
costs. Exact baseline/final Wasm hashes are in the measurement JSON.

This is a deliberate pre-release v1 asset replacement: earlier development frames are not
compatible. Encoder search optimizations alone can continue after adoption without changing
the decoder; changes to decoding rules, dictionaries, or priors require versioned generations.
No released-format migration or backward decoder was added in this experiment.

Validation: `bun run verify-full`, Rust release tests with all features (13 tests), and
Clippy with all targets/features and warnings denied pass. The final SIMD comparison
benchmark verifies 10,542 round-trips; codec-only mode was also verified on the scalar milestone. SIMD and scalar frames match byte for byte on all 3,514
documents and cross-decode in both directions. Wasm-target Clippy also passes. Retraining
reproduces all 48 dictionary/prior files byte for byte; rebuilding reproduces the measured Wasm SHA-256. The regression test
checks that interleaved documents cannot change prior frames or later compression results.
