/**
 * Compact byte-array embedding for generated language modules (standard base64, pure JS —
 * no Buffer/atob so modules load identically in Node, Bun, and browsers).
 */

const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_VALUES = new Int8Array(128).fill(-1);
for (let i = 0; i < 64; i++) BASE64_VALUES[BASE64.codePointAt(i)!] = i;

export function toBase64(bytes: Uint8Array): string {
  const out: string[] = [];
  let i = 0;
  for (; i + 3 <= bytes.length; i += 3) {
    const bits = (bytes[i]! << 16) | (bytes[i + 1]! << 8) | bytes[i + 2]!;
    out.push(BASE64[(bits >>> 18) & 63]!, BASE64[(bits >>> 12) & 63]!, BASE64[(bits >>> 6) & 63]!, BASE64[bits & 63]!);
  }
  const tail = bytes.length - i;
  if (tail === 1) {
    const bits = bytes[i]! << 4;
    out.push(BASE64[(bits >>> 6) & 63]!, BASE64[bits & 63]!, '=', '=');
  } else if (tail === 2) {
    const bits = (bytes[i]! << 10) | (bytes[i + 1]! << 2);
    out.push(BASE64[(bits >>> 12) & 63]!, BASE64[(bits >>> 6) & 63]!, BASE64[bits & 63]!, '=');
  }
  return out.join('');
}

export function fromBase64(text: string): Uint8Array {
  let end = text.length;
  while (end > 0 && text[end - 1] === '=') end--;
  const out = new Uint8Array(Math.floor((end * 6) / 8));
  let acc = 0;
  let accBits = 0;
  let o = 0;
  for (let i = 0; i < end; i++) {
    // Code points >= 128 would index past the table (undefined coerces to 0 in `|`), so guard first.
    const code = text.codePointAt(i)!;
    const value = code < 128 ? BASE64_VALUES[code]! : -1;
    if (value < 0) throw new RangeError('invalid base64 in module data');
    acc = (acc << 6) | value;
    accBits += 6;
    if (accBits >= 8) {
      accBits -= 8;
      out[o++] = (acc >>> accBits) & 255;
    }
  }
  return out;
}
