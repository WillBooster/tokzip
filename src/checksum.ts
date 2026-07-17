/**
 * CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320) over the decompressed content of a
 * frame. The same integrity guarantee gzip carries in its trailer: a frame that decodes
 * structurally but to the wrong bytes — a decoder bug, a corrupted dictionary, a bit flip
 * that survives the entropy coder — fails loudly instead of returning silently wrong data.
 */

const CRC_TABLE = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = (c & 1) !== 0 ? 0xED_B8_83_20 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c;
}

/** CRC-32 of `bytes` as an unsigned 32-bit integer. */
export function crc32(bytes: Uint8Array): number {
  let crc = 0xFF_FF_FF_FF;
  for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]!) & 0xFF]! ^ (crc >>> 8);
  // oxlint-disable-next-line unicorn/prefer-math-trunc -- >>> 0 converts to unsigned; Math.trunc would keep the sign
  return (crc ^ 0xFF_FF_FF_FF) >>> 0;
}
