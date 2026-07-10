/**
 * Logarithmic slot codec shared by `small`-mode length and offset coding (DEFLATE-style):
 * values 0–3 are direct slots; above that, two slots per octave with `nb - 1` raw extra bits,
 * where `nb = floor(log2(value))`.
 */

/** Slot count covering values below 2^18 (length coding). */
export const LENGTH_SLOT_COUNT = 36;
/** Slot count covering values below 2^20 (offset coding, 1 MB window). */
export const OFFSET_SLOT_COUNT = 40;

/** Highest value representable with `slotCount` slots. */
export function maxSlotValue(slotCount: number): number {
  const lastSlot = slotCount - 1;
  if (lastSlot < 4) return lastSlot;
  const nb = ((lastSlot - 4) >>> 1) + 2;
  const high = (lastSlot - 4) & 1;
  // Last slot filled with all-ones extra bits.
  return (1 << nb) | (high << (nb - 1)) | ((1 << (nb - 1)) - 1);
}

export function slotOf(value: number): number {
  if (value < 4) return value;
  const nb = 31 - Math.clz32(value);
  return 4 + 2 * (nb - 2) + ((value >>> (nb - 1)) & 1);
}

export function extraBitsOf(slot: number): number {
  return slot < 4 ? 0 : ((slot - 4) >>> 1) + 1;
}

export function extraValueOf(value: number, slot: number): number {
  return slot < 4 ? 0 : value & ((1 << extraBitsOf(slot)) - 1);
}

/** Reconstructs a value from its slot and raw extra bits. */
export function valueOfSlot(slot: number, extra: number): number {
  if (slot < 4) return slot;
  const nb = ((slot - 4) >>> 1) + 2;
  const high = (slot - 4) & 1;
  return (1 << nb) | (high << (nb - 1)) | extra;
}
