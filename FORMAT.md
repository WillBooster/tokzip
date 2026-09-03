# tokzip frame format (version 1)

The format is the codec: the token grammar, the range coder, the language dictionaries, and the
model priors together. Any change to any of them — a retrained or added dictionary as much as a
new model — is a new version, and `compress` writes only the newest while `decompress` reads
every version released before it. Decoders reject an unknown version instead of misdecoding
it.

The reference implementation is `rust/crates/tokzip`. All multi-byte integers are little-endian;
varints are LEB128 (7 payload bits per byte, bit 7 = continue, canonical: a multi-byte varint
never ends in a zero byte).

## Frame

```
frame := header sizeVarint crc32 body
```

- `header`: one byte — magic `1101` in the high nibble, then two version bits (1; the value 3
  is reserved for an extension byte that would follow), then two layout bits: 0 =
  single-segment, 1 = multi-segment, 2 = blocked, 3 = stored.
- `sizeVarint`: decompressed length in bytes. For coded frames decoders reject a length above
  8192 × the body length before allocating anything — and, unless the frame is blocked, above
  4 MiB — and reject a frame whose coded body needs more than a fixed synthetic-padding budget
  past its end while decoding.
- `crc32`: IEEE CRC-32 of the decompressed content; decoders verify it before returning.
- `body`: stored → the content itself (exactly `size` bytes). Blocked → the block sequence
  (§1). Otherwise a segment table (§2) then the range-coded stream (§4).

The content is a UTF-8 string; the format carries no content type.

Stored frames are the encoder's fallback whenever coding would not be strictly smaller or the
encoder's own decode check fails; empty content is always stored.

## 1. Blocks

Content above 4 MiB is coded as consecutive blocks of 4 MiB (the last one shorter), so the
coder's working set is bounded by the block size; a blocked frame whose `size` fits one block
is a structural error. Each block is coded independently: its own language segments and
models, and tokens never reach into an earlier block.

```
block := blockLayout payloadLenVarint payload
```

- `blockLayout`: one byte with the frame header's layout values — 0 single-segment, 1
  multi-segment, 3 stored (2 and anything above 3 are a structural error).
- `payload`: stored → the block's content (exactly the block length). Otherwise the block's
  segment table (§2) then its range-coded stream (§4), laid out exactly as a single-block body.

The encoder stores a block whose coded payload would not be strictly smaller or fails its
decode check — and, without coding the rest, a block longer than 256 KiB whose first 256 KiB
does not shrink when coded on its own — and falls back to a stored frame when the blocked frame
as a whole would not be smaller. Trailing bytes after the last block are a structural error.

## 2. Segments

The encoder splits the content (or each block) into language segments (contiguous, covering the whole
content). Single segment: one byte, the language id. Multi-segment (layout 1): a varint
segment count (at most 2^20; decoders reject more), then per segment a language-id byte and a
varint length; lengths are non-zero and sum to `size` (the block length in a blocked frame).

Language ids (format identity — the dictionaries and priors they name are part of the codec).
Each language belongs to a model group (§5):

| id  | language   | group    | dictionary suffix |
| --- | ---------- | -------- | ----------------- |
| 0   | text       | prose    | 128 KB            |
| 1   | en-US      | prose    | 128 KB            |
| 2   | ja-JP      | japanese | 128 KB            |
| 3   | html       | code     | 64 KB             |
| 4   | css        | code     | 64 KB             |
| 5   | javascript | code     | 64 KB             |
| 6   | typescript | code     | 64 KB             |
| 7   | c          | code     | 64 KB             |
| 8   | cpp        | code     | 64 KB             |
| 9   | csharp     | code     | 64 KB             |
| 10  | dart       | code     | 64 KB             |
| 11  | haskell    | code     | 64 KB             |
| 12  | java       | code     | 64 KB             |
| 13  | jsp        | code     | 64 KB             |
| 14  | php        | code     | 64 KB             |
| 15  | python     | code     | 64 KB             |
| 16  | ruby       | code     | 64 KB             |
| 17  | rust       | code     | 64 KB             |
| 18  | zig        | code     | 64 KB             |
| 19  | zh-CN      | chinese  | 128 KB            |
| 20  | zh-TW      | chinese  | 128 KB            |

Each language's dictionary is the shared wrapper (`dict/wrapper.bin`) followed by its trained
suffix (`dict/<language>.bin`); its models start from `priors/<language>.bin` (§5).

## 3. Window

Tokens address a source by distance `d ≥ 1` from the current position `p`: `d ≤ p` reaches
into the already-decoded content, larger distances continue into the active segment's
dictionary as if it preceded the content (`dict[D − (d − p)]`, `D` = dictionary length). A
copy runs forward one byte at a time, so it may overlap its own output and may cross from the
dictionary into the content. Tokens never cross a segment boundary.

## 4. Coded stream

One LZMA-style adaptive binary range coder (32-bit range, 11-bit probabilities, adaptation
shift 5, renormalization at 2^24) codes the whole body (one block's payload in a blocked
frame) across all of its segments. The encoder drops the always-zero first output byte, ends
the stream on the value of the final interval with the most trailing zero bytes, and trims
trailing zero bytes; the decoder feeds zeros past the end of the body and rejects a body that
has bytes it never read.

Coder state shared across segments: the 12-state LZMA state machine and the four most recent
distances (`reps`, stored as distance − 1, initially 0). At every segment start, a rep whose
distance exceeds `p + D` of the new segment resets to 0 (distance 1). Each segment swaps in its
language's adaptive models; a language's models persist across all of its segments in one block.

Token grammar per position (bits are coded with the model node named in brackets; `s` is the
state):

- `[is_match s] = 0` → **literal**: 8 bits MSB-first through the literal tree of the context
  pair (class of the previous byte, class of the byte before it; §5). After a match (`s ≥ 7`) the first bits are coded through the shared
  matched-literal trees, predicted by the byte at `reps[0] + 1`, until the first mismatch.
- `[is_match s] = 1`, `[is_rep s] = 0` → **explicit match**:
  - `[is_dict s] = 0`: length (`LEN` model) then distance − 1 (`HIST_DIST` model).
  - `[is_dict s] = 1`: length (`DICT_LEN` model) then the absolute dictionary offset
    (`DICT_OFF` model); the distance is `p + D − offset`.
  - `reps` = [distance − 1, reps[0], reps[1], reps[2]].
- `[is_match s] = 1`, `[is_rep s] = 1` → **rep match**: `[is_rep_g0 s] = 0` selects `reps[0]`
  (`[is_rep0_long s] = 0` is a single-byte short rep with no length); otherwise `is_rep_g1` /
  `is_rep_g2` select `reps[1..3]`, which move to the front. Then a length (`REP_LEN` model).

Lengths: `choice` bit → 3-bit tree (2–9), `choice2` bit → 3-bit tree (10–17), else 8-bit tree
(18–273). Distances/offsets: 6-bit slot tree per length state (`min(len − 2, 3)`), then for
slots ≥ 4 the LZMA footer bits — reverse bit trees for slots < 14 (`spec_pos`), direct bits plus
a 4-bit reverse `align` tree beyond.

## 5. Models and priors

All probabilities of a language live in one flat array (layout in `lz.rs`: `is_match`,
`is_rep`, `is_rep_g0`, `is_rep_g1`, `is_rep_g2`, `is_rep0_long`, `is_dict` × 12 states;
`LEN`, `REP_LEN`, `DICT_LEN` × 274; `HIST_DIST`, `DICT_OFF` × 386; 128 × 4 literal context
pairs × 256 plain-tree nodes; 512 shared matched-literal nodes). The literal context of a
position is the pair (class of the previous byte, class of the byte before it), each byte
continuing into the dictionary before the content start and 0 where neither exists.
`priors/<language>.bin` holds the 256-entry previous-byte → class table (values < 128), the
256-entry second-previous-byte → class table (values < 4), then every node's initial
probability quantized to 8 bits (`p11 = (q << 3) | 4`). All are trained offline
(`bun run train`) and are format identity: a retrain changes the coded stream.

The class tables and the literal trees (everything from the literal context pairs on) are
shared by the languages of a model group — prose (`text`, `en-US`), Japanese, Chinese, code —
and trained on the group's pooled literal statistics; a literal node whose trained value would
have saved fewer than 4 bits on the training data stays at the flat probability 1/2. The nodes
before the literal trees are per language.

The module embeds these values in a packed form that is not part of the format (see
`rust/crates/tokzip/build.rs` and `pack.rs`): each group's literal part once, with every flat
subtree skipped; each language's own nodes verbatim; and each dictionary suffix coded by the
codec itself as one segment with the language's models and no dictionary, decoded on first
use.
