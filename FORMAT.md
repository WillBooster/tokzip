# tokzip frame format (version 0)

Pre-release: the format changes freely and no compatibility with previous frames is kept until
it is fixed as version 1. Decoders reject any other version instead of misdecoding it.

The reference implementation is `rust/crates/tokzip`. All multi-byte integers are little-endian;
varints are LEB128 (7 payload bits per byte, bit 7 = continue, canonical: a multi-byte varint
never ends in a zero byte).

## Frame

```
frame := 0xD0 flags sizeVarint crc32 body
```

- `0xD0`: magic `1101` in the high nibble, version 0 in the low nibble.
- `flags`: bit 0 = content type (0 = UTF-8 string, 1 = bytes); bit 1 = stored; bit 2 =
  multi-segment (never together with stored). Other bits are a structural error.
- `sizeVarint`: decompressed length in bytes (decoders cap it at 256 MiB).
- `crc32`: IEEE CRC-32 of the decompressed content; decoders verify it before returning.
- `body`: stored → the content itself (exactly `size` bytes). Otherwise a segment table then
  the range-coded stream (§3).

Stored frames are the encoder's fallback whenever coding would not be strictly smaller or the
encoder's own decode check fails; empty content is always stored.

## 1. Segments

The encoder splits the content into language segments (contiguous, covering the whole
content). Single segment: one byte, the language id. Multi-segment (flag bit 2): a varint
segment count, then per segment a language-id byte and a varint length; lengths are non-zero
and sum to `size`.

Language ids (format identity — the dictionaries and priors they name are part of the codec):

| id  | language   |
| --- | ---------- |
| 0   | text       |
| 1   | en-US      |
| 2   | ja-JP      |
| 3   | html       |
| 4   | css        |
| 5   | javascript |
| 6   | typescript |

Each language's dictionary is the shared wrapper (`dict/wrapper.bin`) followed by its trained
suffix (`dict/<language>.bin`); its models start from `priors/<language>.bin` (§4).

## 2. Window

Tokens address a source by distance `d ≥ 1` from the current position `p`: `d ≤ p` reaches
into the already-decoded content, larger distances continue into the active segment's
dictionary as if it preceded the content (`dict[D − (d − p)]`, `D` = dictionary length). A
copy runs forward one byte at a time, so it may overlap its own output and may cross from the
dictionary into the content. Tokens never cross a segment boundary.

## 3. Coded stream

One LZMA-style adaptive binary range coder (32-bit range, 11-bit probabilities, adaptation
shift 5, renormalization at 2^24) codes the whole body across all segments. The encoder drops
the always-zero first output byte and trims trailing zero bytes; the decoder feeds zeros past
the end of the body and rejects a body that has bytes it never read.

Coder state shared across segments: the 12-state LZMA state machine and the four most recent
distances (`reps`, stored as distance − 1, initially 0). Each segment swaps in its language's
adaptive models; a language's models persist across all of its segments in one frame.

Token grammar per position (bits are coded with the model node named in brackets; `s` is the
state):

- `[is_match s] = 0` → **literal**: 8 bits MSB-first through the literal tree of the previous
  byte's class (§4). After a match (`s ≥ 7`) the first bits are coded through the shared
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

## 4. Models and priors

All probabilities of a language live in one flat array (layout in `lz.rs`: `is_match`,
`is_rep`, `is_rep_g0`, `is_rep_g1`, `is_rep_g2`, `is_rep0_long`, `is_dict` × 12 states;
`LEN`, `REP_LEN`, `DICT_LEN` × 274; `HIST_DIST`, `DICT_OFF` × 386; 32 literal classes × 256
plain-tree nodes; 512 shared matched-literal nodes). `priors/<language>.bin` holds the
256-entry previous-byte → class table followed by every node's initial probability quantized
to 8 bits (`p11 = (q << 3) | 4`). Both are trained offline (`bun run train`) and are format
identity: a retrain changes the coded stream.
