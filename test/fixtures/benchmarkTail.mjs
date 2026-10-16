import assert from 'node:assert/strict';
import { readFileSync, watch } from 'node:fs';

const file = process.env.TOKZIP_BENCHMARK_EVENTS;
assert(file);
let offset = 0;
const watcher = watch(file, flush);
flush();
process.on('SIGTERM', () => watcher.close());

function flush() {
  const text = readFileSync(file, 'utf8');
  for (const line of text.slice(offset).split('\n').filter(Boolean)) {
    const event = JSON.parse(line);
    process.stdout.write(JSON.stringify({ event: { request: { url: event.url } }, cpuTime: event.cpuMs }) + '\n');
  }
  offset = text.length;
}
