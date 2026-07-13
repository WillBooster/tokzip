# tokzip text frame format (version 2)

This document is the normative wire-format specification for tokzip payloads. It is
self-contained so the format can be ported to other implementation languages. The reference
implementation lives under `src/`.

A tokzip payload is a single **text frame**: a safe-ASCII string generated in one pass, with
no binary intermediate and no base64 stage. Trailing characters after the frame are a
structural error. All structural errors MUST be reported as a typed decode error
(`TokzipDecodeError` in the reference implementation); valid-looking corruption MAY decode to
wrong output without throwing (there is no integrity checksum in v2; a flag bit is reserved).

## 1. Alphabets

### 1.1 Radix-64 alphabet (header, `fast` bodies)

The base64url character set, indexed 0–63:

```
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_
```

Every radix-64 field is a whole number of 6-bit characters. Characters outside the alphabet
are a structural error.

### 1.2 Radix-85 alphabet (`small` bodies)

Printable ASCII 0x21–0x7E excluding the nine unsafe characters `"` `\` `` ` `` `$` `<` `>`
`&` `'` `%`, indexed 0–84 in ascending code-point order:

```
!#()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{|}~
```

Both alphabets are JSON-string-safe, template-literal-safe, and HTML-attribute-safe; the
radix-64 alphabet is additionally URL-safe.

## 2. Numbers

### 2.1 Radix-64 varint

Little-endian groups of 5 payload bits per character; character value = `payload | 0x20` when
another group follows, `payload` otherwise. Maximum 7 characters (35 bits). **Canonical form
is required**: a multi-character varint whose final group is zero is a structural error, and
decoders MUST reject varints longer than 7 characters.

### 2.2 Bit-level varint (`small` bodies)

Inside a `small` bitstream: 8-bit groups, 7 payload bits little-endian first, MSB of the
group = continue flag. Maximum 8 groups.

### 2.3 Raw byte packing (pinned alignment)

Raw bytes are bit-packed into radix-64 characters at fixed alignment, MSB-first: 3 bytes → 4
characters. Tail rule (normative): 1 trailing byte → 2 characters (byte in the high 8 of 12
bits), 2 trailing bytes → 3 characters (bytes in the high 16 of 18 bits). Padding bits are
zero on encode and ignored on decode.

## 3. Frame layout

```
[0] magic|version   radix-64 char; value 0b110_010 (50, char 'y') for v2.
[1] language id     radix-64 char; 0–63.
[2] flags           radix-64 char:
                      bits 1:0  shipped mode: 0 stored, 1 fast, 2 small; 3 is invalid
                      bit  2    input type: 0 string (UTF-8), 1 bytes
                      bits 5:3  reserved; encoders write 0, decoders reject non-zero
[3…] decompressed size   radix-64 varint (bytes of the decompressed payload)
[…]  body                per shipped mode (sections 5–7)
```

- A magic/version char with the correct magic (high 3 bits `0b110`) but a different version
  is "unknown version"; anything else is "bad magic".
- The declared size MUST be validated against `maxOutputSize` (implementation default
  64 MiB) **before** any allocation.
- The return type of decompression follows the input-type flag. String frames MUST be decoded
  as UTF-8 in fatal mode: invalid UTF-8 is a structural error, never U+FFFD insertion.
  (On the encode side, JS lone surrogates become U+FFFD per WHATWG `TextEncoder`; byte-exact
  callers use the bytes path.)
- Frames are single: any character after the body is a structural error.

## 4. Language ids (v2 allocation, unchanged from v1)

| id  | language            |     | id  | language   |
| --- | ------------------- | --- | --- | ---------- |
| 0   | none (wrapper only) |     | 11  | javascript |
| 1   | text                |     | 12  | php        |
| 2   | c                   |     | 13  | python     |
| 3   | cpp                 |     | 14  | ruby       |
| 4   | csharp              |     | 15  | rust       |
| 5   | css                 |     | 16  | typescript |
| 6   | dart                |     | 17  | zig        |
| 7   | haskell             |     | 18  | en-US      |
| 8   | html                |     | 19  | ja-JP      |
| 9   | java                |     | 20  | zh-CN      |
| 10  | jsp                 |     | 21  | zh-TW      |

Id 22 is reserved for XML (deferred). Ids 23–63 are unallocated; decoders treat unknown ids
as a structural error on non-stored frames.

**Stored frames**: encoders write language id 0; decoders accept and ignore any id (stored
frames need zero registration).

## 5. Stored body

The declared number of bytes, packed per §2.3. The body character count MUST equal exactly
`packedLength(size)`; anything shorter is truncation, anything longer is trailing characters.
Total frame overhead is exactly header (3) + size varint — the format never expands input
beyond that plus the 4/3 packing tax.

## 6. Dictionaries and the shared LZ token model

Both modes serialize the same token semantics: literal runs, history matches, dictionary
matches, and rep matches.

- **Decoupled dual dictionary.** The shared wrapper dictionary (core) occupies offsets
  `0 … wrapperLen-1`; a registered language's suffix follows contiguously. Language id 0 is
  wrapper-only. The dictionary never slides and is never prepended to history. Dictionary
  match offsets are absolute start positions in this assembled space; a dictionary match MUST
  lie entirely within it.
- **History matches** address the produced output: distance `d ≥ 1`, `d ≤` bytes produced so
  far. Overlap-copy is required (`d < length` copies bytes produced by the same match,
  byte-serially from the start).
- **Rep-offset cache (normative).** Four history distances, initialized to `[1, 2, 3, 4]`
  (most recent first). On a rep match `repN`: the entry moves to the front. On an explicit
  history match: its distance is inserted at the front and the last entry drops out.
  Literal runs and dictionary matches do not modify the cache. Rep distances are
  bounds-checked like explicit ones.
- **Minimum lengths** (bases of the length coding): rep matches 2; explicit history and
  dictionary matches 4.
- **Windows (normative per mode):** `fast` 256 KiB (2^18), `small` 1 MiB (2^20).
- Encoders MUST split matches longer than 262145 bytes (= `small` length-slot bound + 2);
  `fast` length varints could represent more, but the cap is format-wide for cross-mode
  token-list compatibility.
- **Cost-based acceptance** is normative as a guarantee, not an algorithm: an encoder MUST
  NOT emit a frame larger than the stored frame of the same input (see §8); the reference
  encoder only accepts matches whose exact output cost beats the literal encoding of the same
  bytes (in `fast`: rep matches become profitable at 2–3 bytes, explicit matches at 4–5
  depending on offset width, chosen by a greedy parse; in `small`: exact static-table bit
  prices drive a shortest-path optimal parse with path-carried rep state, falling back to a
  bounded price-aware lazy parse beyond the encoder's input bound).

## 7. `fast` body: char-aligned radix-64 token stream

A sequence of tokens; each token is a **tag char** followed by fields. Tag value =
`kind << 3 | payload`. All 8 kinds are allocated:

| kind | meaning          |
| ---- | ---------------- |
| 0    | literal-64 run   |
| 1    | literal-raw run  |
| 2    | history match    |
| 3    | dictionary match |
| 4–7  | rep0–rep3 match  |

### 7.1 Literal runs (kinds 0–1)

`payload` 0–6 → run length 1–7; `payload` 7 → length = 8 + varint (§2.1), and the varint
immediately follows the tag (before the body, so decoding stays single-pass).

- **literal-64** body: one radix-64 char per byte, indexing the language's **top-64 literal
  charset** (a 64-byte table shipped with each language module; the id-0 charset ships in
  core). The decoder maps char value → `top64[value]`.
- **literal-raw** body: run bytes packed per §2.3.

### 7.2 History / dictionary matches (kinds 2–3)

`payload = width << 2 | lencode`.

- `width` 0: offset field is 2 chars (12 bits); `width` 1: 3 chars (18 bits). Fields are
  MSB-first. For history matches the field holds `distance - 1`; for dictionary matches the
  absolute dictionary start.
- `lencode` 0–2 → length = 4 + lencode (4–6); `lencode` 3 → length = 7 + varint, the varint
  following the offset field.
- Encoders use the short width whenever the field value fits in 12 bits (both widths decode
  identically).

### 7.3 Rep matches (kinds 4–7)

`payload` 0–6 → length = 2 + payload (2–8); `payload` 7 → length = 9 + varint following the
tag.

### 7.4 Decoding

Tokens are decoded until exactly the declared size has been produced; producing beyond it,
running out of characters, or leftover characters afterwards are structural errors, as are
out-of-bounds history distances and dictionary ranges.

## 8. `small` body: separated streams, static entropy coding, radix-85

The entire body is one MSB-first bitstream emitted through the fused Z85-style writer: each
32-bit word becomes 5 radix-85 chars (§1.2), most-significant digit first. The final partial
word is zero-padded; a body whose character count is not a multiple of 5, that contains
more words than the minimal count for the used bits, or whose padding bits are non-zero,
is a structural error (frames have a single canonical encoding). Because of §9, decoders
also reject any non-stored body that is not strictly smaller than the stored body of the
declared size — this additionally bounds decode-side allocations by the declared output
size, which is validated against `maxOutputSize` first.

### 8.1 Bitstream layout

```
[3 bits]  stream modes: bit2 literals, bit1 tokens, bit0 offsets (1 = Huffman, 0 = raw)
[varint]  token count                        (§2.2)
[varint]  literal-stream bit length
[varint]  token-stream bit length
[bits]    literal stream
[bits]    token stream
[bits]    offset stream
[padding] zeros to the 32-bit boundary
```

The three streams are decoded with independent cursors; after decoding, the literal and token
cursors MUST land exactly on their recorded boundaries.

### 8.2 Static context-modeled entropy coding

Each language module ships **static canonical Huffman code lengths** for three alphabets,
**one table per context** (all contexts are decoder-derivable state, so frames carry nothing
extra):

- **literal** stream: byte values, 256 symbols. Context = the module's trained **literal
  context class** of the previous decoded output byte (byte value 0 when no byte has been
  produced yet — including at the start of a run after nothing but position 0). The module
  ships a 256-entry class map (`litContext`) and `litClassCount` tables (1–64 classes).
- **token** stream: `kind × 36 + lengthSlot`, 252 symbols, kinds:
  0 literal-run, 1 history, 2 dictionary, 3–6 rep0–rep3. Context = the **previous token's
  kind** (7 contexts; the first token of a frame uses kind 0, literal-run).
- **offset** stream: offset slots, 40 symbols. Context = the match kind consuming the offset:
  0 history, 1 dictionary (2 contexts).

Every context's code lengths are ≤ 12 and MUST form a complete code (Kraft sum exactly 1)
over the used alphabet, validated at registration; decoders use a 4096-entry single-lookup
table per context. Canonical assignment: codes ordered by (length, symbol index). A per-frame
**raw mode** per stream (mode bit 0) encodes symbols as fixed-width integers instead,
ignoring contexts: literals 8 bits, token symbols 8 bits, offset slots 6 bits — this covers
degenerate/empty streams and inputs whose symbols lack codes in the shipped tables.

The literal context is the previous **output** byte, which for a conforming encoder is
simply the previous input byte — literal pricing is therefore tokenization-independent and
exact even inside an optimal parse.

### 8.3 Slot coding (lengths and offsets)

For a value `v ≥ 0`: values 0–3 are direct slots 0–3 with no extra bits; otherwise with
`nb = floor(log2 v)`, slot = `4 + 2(nb-2) + bit(nb-1 of v)` and `nb - 1` raw extra bits (the
low bits of `v`), written MSB-first immediately after the symbol in the same stream.

- **Token symbols** carry the length: for literal runs `v = runLength - 1`, for matches
  `v = length - 2`; 36 slots bound `v < 2^18`.
- **Offset slots**: for history matches `v = distance - 1`, for dictionary matches
  `v = start`; 40 slots bound `v < 2^20` (the 1 MiB window). Rep matches and literal runs
  write nothing to the offset stream.

### 8.4 Decoding

`tokenCount` tokens are read from the token stream. Literal-run tokens then read `runLength`
symbols from the literal stream; history/dictionary tokens read one slot (+extras) from the
offset stream; rep tokens read nothing further. Bounds rules match §7.4, plus the two cursor
boundary checks of §8.1 and the minimal-padding check of §8.

## 9. Auto-downgrade (normative)

`small` is a maximum effort, not a promise. The encoder compares complete frames — small vs
fast vs stored — **analytically** (every token's `fast` char cost and the stored size are
arithmetic; the `small` bit total is exact because the tables are static). The fast
candidate is the cheaper of the small-parsed token list re-priced in `fast` chars and a pure
`fast` parse of the same input, so `small` output is never larger than `fast` output.
Only the winning frame is emitted; ties choose the simpler encoding
(stored ≺ fast ≺ small). If any token exceeds `fast`'s representable ranges (offset ≥ 2^18,
dictionary start ≥ 2^18, or explicit match shorter than 4), the fast candidate is ineligible
and the comparison is small vs stored. The header records what shipped; decoders branch only
on that. `fast`-mode encoding performs the same fast-vs-stored comparison. Consequently
output never expands beyond header + size varint + stored body.

Downgrade sizing MUST be exact (not estimated) so the shipped mode is deterministic across
conforming implementations given the same token list.

## 10. Language modules

A module ships: language id + name, dictionary suffix bytes, the 64-byte top-64 literal
charset, the literal context class map + class count, and the three per-context code-length
arrays of §8.2. Tables are validated at registration
(complete code, alphabet sizes, charset length 64); the dictionary match index is built
lazily on first compress per language and cached per process (idempotent, re-entrant).
Compressing with an explicitly requested unregistered language throws; decoding a non-stored
frame with an unregistered id is a structural error.

**Module data is part of the codec identity.** Non-stored frames reference the registered
module's dictionary bytes, top-64 charset, context map, and code lengths implicitly — the
frame carries only the language id. A non-stored frame therefore decodes correctly only with the exact
trained module data its encoder used; retraining dictionaries or tables is a breaking change
for persisted non-stored frames even though the wire format version is unchanged (stored
frames are unaffected). Deployments that persist frames across library upgrades must pin the
library version or re-encode; see the tracking issue on versioned module assets.

## 11. Conformance vectors

Executable vectors live in `test/conformance.test.ts` and `test/roundtrip.test.ts`; the
normative list mirrors the design issue:

empty input (exact frame `yAAA`); tiny stored frame (exact overhead bound); history /
dictionary / rep / overlap-copy matches; 12-bit vs 18-bit offset forms; single-symbol and
degenerate streams (raw stream modes); downgrade tie and determinism; invalid table; unknown
id on non-stored frames; stored frame with nonzero id (decodes); unknown version; reserved
flag bit set; truncated header / token / stream; trailing characters; non-alphabet character;
non-canonical varint; `maxOutputSize` exceeded; invalid UTF-8 body; non-stored body not
smaller than the stored body (e.g. a size-0 `small` frame); literal-64 vs literal-raw
runs including 1- and 2-byte raw tails; downgrade with fast-ineligible tokens.
