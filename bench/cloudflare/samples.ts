import samples from './samples.json' with { type: 'json' };

const answer = samples['mixed-answer-20k'];
export const SAMPLES: Record<string, string> = {
  ...samples,
  'json-response': JSON.stringify({
    text: answer,
    reasoning: 'The answer combines prose and source code.\n'.repeat(40),
    toolCalls: [{ id: 'call-1', name: 'read_file', input: { path: 'example.ts' } }],
    finishReason: 'stop',
  }),
  'mixed-1m': answer.repeat(Math.ceil((1024 * 1024) / new TextEncoder().encode(answer).length)),
};
