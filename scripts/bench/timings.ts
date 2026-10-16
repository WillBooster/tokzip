export interface Timings {
  compress: number[];
  decompress: number[];
}

interface Percentiles {
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

interface TimingSummary {
  compressMBps: number;
  decompressMBps: number;
  compressMs: Percentiles;
  decompressMs: Percentiles;
}

export function summarizeTimings(bytes: number, timings: Timings): TimingSummary {
  return {
    compressMBps: throughput(bytes, timings.compress),
    decompressMBps: throughput(bytes, timings.decompress),
    compressMs: percentiles(timings.compress),
    decompressMs: percentiles(timings.decompress),
  };
}

function throughput(bytes: number, times: number[]): number {
  return bytes / times.reduce((sum, time) => sum + time, 0) / 1000;
}

function percentiles(times: number[]): Percentiles {
  const sorted = times.toSorted((a, b) => a - b);
  const at = (fraction: number): number => sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]!;
  return { p50: at(0.5), p95: at(0.95), p99: at(0.99), max: sorted.at(-1)! };
}
