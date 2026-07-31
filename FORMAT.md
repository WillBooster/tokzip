# tokzip frame format (version 2)

This document is the normative wire-format specification for tokzip payloads. It is
self-contained so the format can be ported to other implementation languages. The reference
implementation lives under `src/`.

A tokzip payload is a single frame in one of two containers sharing the same coding model:
the **text frame** (§2–§7) — a safe-ASCII string — and the **binary frame** (§8) — the same
stream at 8 bits per byte. Trailing characters (or bytes) after the frame are a structural
error. All structural errors MUST be reported as a typed decode error (`TokzipDecodeError`
in the reference implementation). Every frame carries a CRC-32 of its decompressed content,
so corruption that survives the structural checks is still caught: decoders MUST verify the
checksum before returning output.

Version 2 replaced version 1 wholesale: the `fast` mode and its radix-64 token stream are
gone, and the `small` body is now a single adaptive binary range-coded stream whose models
start from trained per-language priors. The format is pre-release and still evolving: a
version bump invalidates all previously written payloads (decoders reject other versions as
"unknown version", never misdecode them), and no cross-version compatibility is provided.

## 1. Alphabets and numbers

### 1.1 Radix-64 alphabet (header, stored bodies)

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

### 1.3 Radix-64 varint

Little-endian groups of 5 payload bits per character; character value = `payload | 0x20`
when another group follows, `payload` otherwise. Maximum 7 characters (35 bits).
**Canonical form is required**: a multi-character varint whose final group is zero is a
structural error, and decoders MUST reject varints longer than 7 characters.

### 1.4 Byte varint (binary container, streams)

Little-endian groups of 7 payload bits per byte, bit 7 = continue flag. Maximum 5 bytes
(35 bits) unless a field says otherwise. Canonical form is required as in §1.3.

### 1.5 Raw byte packing (stored bodies, text container)

3 input bytes → 4 radix-64 chars (big-endian 6-bit groups); a 1- or 2-byte tail packs into
2 or 3 chars. Encoders write zero in the tail chars' unused low bits; decoders ignore them
(the content CRC still covers the decoded bytes). `packedRawLength(n) = 4·floor(n/3) +
(0, 2, 3)[n mod 3]`.

### 1.6 CRC-32

The IEEE 802.3 polynomial (as in gzip), over the frame's decompressed content followed by
one type byte: 0x00 for string-typed frames, 0x01 for bytes-typed frames (so retyping
byte-identical content fails the checksum). Text frames store it as 6 radix-64 chars,
little-endian 6-bit groups, top 4 bits of the last group zero (non-zero is a structural
error); binary frames and stream blocks store 4 little-endian bytes.

## 2. Text container

```
frame := magic language flags sizeVarint crc32 body
```

- `magic` (1 char): radix-64 value 0b110010 — 3-bit magic `110`, 3-bit version 2. A
  different version under the same magic MUST be rejected as "unknown version"; anything
  else as "bad magic".
- `language` (1 char): the language id (§6). Stored frames always write id 0 and decoders
  ignore it there.
- `flags` (1 char): bits 1:0 = mode (0 stored, 2 small; 1 and 3 are structural errors);
  bit 2 = content type (0 string/UTF-8, 1 bytes); bit 3 = fenced dictionary extension
  (§6.1); bits 5:4 reserved, MUST be zero.
- `sizeVarint`: the decompressed content length in bytes (§1.3).
- `crc32`: §1.6.
- `body`: §3 (stored) or §4 (small).

**Canonical frames.** A stored body's length is exactly `packedRawLength(size)`. A small
body MUST be strictly smaller than the stored body of the same content (encoders downgrade
to stored otherwise, choosing stored on ties); a small frame that is not is a structural
error ("non-canonical frame"). String-typed output MUST decode as valid UTF-8 (fatal, BOM
preserved); lone surrogates were replaced with U+FFFD at compress time per WHATWG
TextEncoder.

## 3. Stored body

The content bytes in raw packing (§1.5). Requires no language registration.

## 4. Small body

The small body is one adaptive binary range-coded stream (§4.1) over the token model
(§4.3), wrapped for the text container as radix-85 (§4.2).

### 4.1 Range coder

An LZMA-style binary range coder with 32-bit range and 11-bit probabilities:

- State: `range` (32-bit, initially 0xFFFFFFFF), `low` (33-bit accumulator with carry,
  initially 0), and on the decoder `code`, initialized by reading 5 bytes — the first MUST
  be 0x00 (the encoder's initial cache byte; anything else is a structural error).
- Coding bit `b` with probability `p` = P(b = 0) out of 2048: `bound = (range >> 11) · p`;
  bit 0 → `range = bound`; bit 1 → `low += bound`, `range −= bound`. After coding, the
  model adapts: bit 0 → `p += (2048 − p) >> 5`; bit 1 → `p −= p >> 5`.
- Renormalization: while `range < 2^24`, the encoder shifts the top byte out of `low`
  (propagating carries into already-buffered bytes); the decoder shifts a byte into
  `code`; both then `range <<= 8`.
- Direct bits (probability ½, no adaptation), MSB first: `range >>= 1`; encoder: bit 1 →
  `low += range`; decoder: `code ≥ range` → bit 1, `code −= range`.
- Bit trees code an n-bit symbol MSB-first through a 1-based heap of `2^n − 1` adaptive
  probabilities (start at node 1; descend to `2·node + bit`).
- Flush: the encoder emits 5 final bytes. The decoder MUST consume exactly the body's
  bytes: reading past the end is "truncated", and unread bytes beyond the channel padding
  (§4.2) are a "trailing" error.

All probabilities start from the language's trained priors (§6), each in [1, 2047]; every
frame codes from a fresh copy, so frames are independent.

### 4.2 Text wrapping

The range-coded bytes are zero-padded to a multiple of 4, and each 4-byte big-endian word
becomes 5 radix-85 chars (most significant digit first). A body length not divisible by 5
and a 5-char group above 2^32 − 1 are structural errors. After the last token, the consumed
byte count MUST be within 3 bytes of the padded length and every remaining byte MUST be
zero ("non-zero padding"). Binary frames carry the raw bytes with no padding: the consumed
count MUST equal the body length exactly. Note that these checks pin the body _length_,
not every bit: the range coder's five flush bytes admit multiple encodings of the same
content (the decoder consumes them without constraining their exact values), so frame
bytes are NOT a unique identity for the content — key dedup/idempotency on the content or
its CRC, never on the frame bytes.

### 4.3 Token model

Output is produced token by token until exactly `size` bytes exist; a token that would
produce past `size` is a structural error, as is a stream that ends early. Models index by
the previous token kind `prevKind` ∈ {LIT 0, HISTORY 1, DICT 2, REP0–REP3 3–6}, initially
LIT. `prevByte` is the last produced byte (0 at the start; with stream history, the last
history byte). Before decoding, a declared size above `bodyBytes × 8 × 5 × 262145` MUST be
rejected ("declared size exceeds body capacity") — even the adaptive probability clamp
cannot code a token in fewer than a fifth of a bit.

Per token:

1. `isMatch[prevKind]` (1 bit): 0 → literal, 1 → match.
2. **Literal**: one byte through the literal model of class `litContext[prevByte]` (§6).
   Each class owns a 768-slot block: a plain 8-bit tree (nodes 1–255 at block offsets
   0–254) and two _matched_ subtrees. If `prevKind ≠ LIT` and `rep0 ≤ produced`, the byte
   at distance `rep0` behind the current output position predicts the byte: while the
   prediction holds, bit `b` is coded at block offset `((1 + matchBit) << 8 | node) − 1`
   (`matchBit` = the predictor's bit at this position); after the first mismatch, coding
   continues in the plain tree. Otherwise the plain tree codes all 8 bits. `prevKind`
   becomes LIT.
3. **Match**: `isRep[prevKind]` (1 bit).
   - **Rep** (1): a 2-bit tree at `repTree[prevKind]` selects rep index r ∈ 0–3; the
     distance is the r-th entry of the rep cache (§4.4). Length via the REP length model.
     `prevKind` becomes REP0 + r.
   - **Explicit** (0): `isDict[prevKind]` (1 bit; 0 → history, 1 → dictionary). The
     **length** is coded first via the HISTORY or DICT length model, then the **offset**
     via the matching offset model with length bucket `min(len − 2, 3)`. History:
     `dist = offset + 1`, which MUST be ≥ 1 and ≤ the produced count (including stream
     history); the match copies `len` bytes from `dist` back (byte-by-byte when
     overlapping). Dictionary: `start = offset`; the match copies
     `dictionary[start, start + len)`, or resolves through the fenced extension (§6.1)
     when the flag is set; out-of-bounds is a structural error. `prevKind` becomes HISTORY
     or DICT.

Lengths and offsets use the slot codec (§5): a 6-bit tree codes the slot (36 length slots,
40 offset slots; a decoded slot beyond its alphabet is "invalid symbol"), then the slot's
extra bits follow as direct bits. Length value = `len − 2` (minimum match length 2;
encoders only emit explicit matches of length ≥ 4). Encoders MUST split matches longer
than 262,145 bytes (`maxSlotValue(36) + 2`). A history distance is the decoded offset
plus one (`dist = offset + 1`; offsets are < 2^20, so distances are at most 2^20 — the
1 MB history window, `SMALL_WINDOW`). Dictionary starts are < 2^20.

Model layout (offsets in probability slots):

| section                                    | offset | count               |
| ------------------------------------------ | ------ | ------------------- |
| isMatch                                    | 0      | 7                   |
| isRep                                      | 7      | 7                   |
| isDict                                     | 14     | 7                   |
| repTree                                    | 21     | 7 × 3               |
| lenTree (REP, HISTORY, DICT)               | 42     | 3 × 63              |
| offTree (HISTORY, DICT × 4 length buckets) | 231    | 8 × 63              |
| literal                                    | 735    | litClassCount × 768 |

### 4.4 Rep-offset cache

Four distances, initially (1, 2, 3, 4). A rep match moves its entry to the front; an
explicit history match pushes its distance at the front (the last entry drops out);
literals and dictionary matches leave the cache unchanged. Encoders and decoders MUST
replay identical updates.

## 5. Slot codec

DEFLATE-style logarithmic slots shared by lengths and offsets: values 0–3 are slots 0–3;
above that, two slots per octave with `nb − 1` extra bits where `nb = floor(log2 v)`:
`slot(v) = 4 + 2·(nb − 2) + ((v >> (nb − 1)) & 1)`, extra bits = the low `nb − 1` bits of
`v`. 36 slots cover values below 2^18 (lengths); 40 slots cover values below 2^20
(offsets).

## 6. Language modules

A language module is codec identity: `(id, name, dictionarySuffix, litContext,
litClassCount, priors)`. The assembled dictionary is the shared wrapper dictionary (shipped
in core, identical for every language) followed by the language's suffix. `litContext` maps
each previous-byte value to one of `litClassCount ≤ 64` literal classes. `priors` hold the
initial probability of every model slot (§4.3), each in [1, 2047]. Registering a module
that differs in any byte from an existing registration of the same id or name MUST be
rejected. Frames reference modules by id; decoding a small frame under a different module
than the encoder used produces garbage that the CRC rejects — it never silently succeeds.

The language-id allocation is normative (unchanged since v1); id 22 is reserved for the
deferred XML module:

| id  | name    | id  | name       | id  | name            |
| --- | ------- | --- | ---------- | --- | --------------- |
| 0   | none    | 8   | html       | 16  | typescript      |
| 1   | text    | 9   | java       | 17  | zig             |
| 2   | c       | 10  | jsp        | 18  | en-US           |
| 3   | cpp     | 11  | javascript | 19  | ja-JP           |
| 4   | csharp  | 12  | php        | 20  | zh-CN           |
| 5   | css     | 13  | python     | 21  | zh-TW           |
| 6   | dart    | 14  | ruby       | 22  | (reserved: xml) |
| 7   | haskell | 15  | rust       |     |                 |

### 6.1 Fenced dictionary extension (flag bit 3)

Documents that embed fenced code blocks (triple-backtick fences, as produced by Markdown
and LLM output) compress better when dictionary matches inside a block can also address
that block language's dictionary. Flag bit 3 enables this per frame:

- **Bit 3 = 0**: every dictionary match addresses the frame language's assembled
  dictionary (such frames are identical to plain unfenced frames).
- **Bit 3 = 1**: the dictionary space is **extended, not switched**. Offsets below the
  frame language's assembled dictionary length keep their plain meaning everywhere. Where
  the **fence state** of the output produced strictly before a match's first output byte
  names a block language other than the frame language, that language's **dictionary
  suffix** is addressed contiguously above the frame dictionary: virtual offset
  `frameDictionaryLength + k` reads suffix byte `k`. The space is contiguous, so a match
  MAY straddle the boundary. All other models (literal classes, contexts, history, reps)
  stay keyed to the frame language. Because the extension is a strict superset of the
  plain space, fenced frames never lose frame-dictionary coverage. The reference encoder
  additionally prices the fenced and plain parses exactly and ships the smaller frame,
  preferring plain on ties.

Fence state is derived from the decoded output itself, so the frame carries nothing beyond
the flag bit. It advances one **completed line** at a time: a line is a byte range ending
at an LF (0x0A); bytes after the last produced LF are pending and never affect state. The
grammar (normative):

- A **fence line** starts at column 0 with three or more backticks (0x60). Its _info
  string_ is the rest of the line with trailing spaces/tabs/CRs (0x20/0x09/0x0D) and
  leading spaces/tabs removed.
- Outside a block, a fence line whose info string contains no backtick **opens** a block,
  recording its backtick count. The _label_ is the info string up to the first space/tab;
  labels are matched ASCII-lowercased against the normative alias table below. An unknown
  or empty label keeps the frame language for the block.

  | language   | labels                                  |
  | ---------- | --------------------------------------- |
  | c          | `c`, `h`                                |
  | cpp        | `cpp`, `c++`, `cc`, `cxx`, `hpp`        |
  | csharp     | `csharp`, `cs`, `c#`                    |
  | css        | `css`                                   |
  | dart       | `dart`                                  |
  | haskell    | `haskell`, `hs`                         |
  | html       | `html`, `htm`                           |
  | java       | `java`                                  |
  | jsp        | `jsp`                                   |
  | javascript | `javascript`, `js`, `jsx`, `mjs`, `cjs` |
  | php        | `php`                                   |
  | python     | `python`, `py`, `python3`               |
  | ruby       | `ruby`, `rb`                            |
  | rust       | `rust`, `rs`                            |
  | typescript | `typescript`, `ts`, `tsx`, `mts`, `cts` |
  | zig        | `zig`                                   |
  | text       | `text`, `txt`, `plain`, `plaintext`     |

- Inside a block, a fence line with at least the opening backtick count and an **empty**
  info string **closes** it; every other line (fence-like or not) is content.
- State transitions take effect at the byte after the fence line's LF. An unclosed block
  extends to the end of the output.

A dictionary match whose range reaches past the frame dictionary length is valid only
where the active block language differs from the frame language, and MUST lie entirely
within `frameDictionaryLength + blockSuffixLength`; decoding such a match requires the
block language's module to be registered (`src/fences.ts` is the single normative
implementation of the grammar, shared by encoder and decoder).

## 7. Decoding guarantees

Decoders MUST never read past the frame, never allocate beyond the declared size (after
the §4.3 capacity check), and either return the exact original content or throw a typed
decode error. Mutated, truncated, and forged frames are covered by fuzz tests; a frame
that decodes without error and passes the CRC is byte-exact.

## 8. Binary container

```
frame := magicByte languageByte flagsByte sizeByteVarint crc32(4 bytes) body
```

`magicByte` = 0x80 | 0b110010 = 0xB2 (bit 7 marks binary; text frames are all-ASCII).
Flags as in §2 with bits 7:4 reserved-zero. Stored bodies are the raw content bytes; small
bodies are the raw range-coded bytes (§4, no padding). Canonicality mirrors §2.

## 9. Stream container

A tokzip stream (produced by `TokzipCompressionStream`) is binary-only:

```
stream := 0xBA languageByte streamFlags block* terminator
```

- `0xBA` = bit 7 | magic 0b111 (bits 6:3) | stream version 2 (bits 2:0).
- `streamFlags`: bit 0 = window carry (blocks decode with previous blocks' output seeded
  as history — matches may reach into it and the literal context chains across the
  boundary); bits 7:1 reserved-zero.
- `block := bodyLenVarint modeByte rawLenVarint crc32 body` — `modeByte` 0 (stored) or 2
  (small: a §4 binary body decoded with the carried history when the flag is set; the
  model state still resets per block). The CRC is the _chained_ CRC-32 of all raw bytes up
  to and including this block, so deletion, reordering, and replay break the chain. A
  non-stored body MUST be strictly smaller than the block's raw length; `rawLen` 0 is a
  structural error.
- `terminator := 0x00 totalRawLenVarint(≤ 8 bytes) crc32` — the total raw byte count and
  final chained CRC, so dropping trailing blocks is detected.

## 10. Module data identity

Trained module data (dictionary bytes, `litContext`, priors) participates in the wire
contract exactly like this document: two implementations interoperate only when their
registered modules are byte-identical per id. The reference modules live in
`src/generated/` and are regenerated by `scripts/train/train.ts`.
