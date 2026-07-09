/**
 * Generates the LLM half of the corpus by driving the `subagent` skill over the prompt
 * matrix: (agent, model) × language × task × size × wrapping, with seeded topics per cell.
 * Outputs are saved verbatim (fences and prose included — that wrapping is exactly what real
 * payloads look like) after validation (non-empty, plausible language, dedupe).
 *
 * Each invocation pins one explicit (agent, model) pair — never `--agent all` — via:
 *   bunx @willbooster-private/agentic-workflows@<version> skills agent \
 *     --agent <agent> --model <model> --cwd . "<prompt>"
 *
 * Generation runs are long (LLM/agent responses can take 1-2 hours; wait patiently rather
 * than assuming failure). Use --limit to bound a batch.
 *
 * Usage: bun scripts/corpus/generateLlm.ts [--limit N] [--dry-run] [<language> ...]
 */
import { spawnSync } from 'node:child_process';
import matrix from './prompt-matrix.json';
import { appendManifest, seededRandom, sizeBucketOf, writeSample } from './shared.ts';

const AGENTIC_WORKFLOWS_VERSION = '3';
const CODE_LANGUAGES = [
  'c',
  'cpp',
  'csharp',
  'css',
  'dart',
  'haskell',
  'html',
  'java',
  'jsp',
  'javascript',
  'php',
  'python',
  'ruby',
  'rust',
  'typescript',
  'zig',
];
const NL_LOCALES = ['en-US', 'ja-JP', 'zh-CN', 'zh-TW'];
const LOCALE_NAMES: Record<string, string> = {
  'en-US': 'English (US)',
  'ja-JP': 'Japanese',
  'zh-CN': 'Simplified Chinese',
  'zh-TW': 'Traditional Chinese (Taiwan)',
};

interface Cell {
  language: string;
  agent: string;
  model: string;
  taskId: string;
  sizeLabel: string;
  wrappingLabel: string;
  prompt: string;
}

function buildCells(languages: string[]): Cell[] {
  const cells: Cell[] = [];
  const random = seededRandom(0xC0_FF_EE);
  const pick = <T>(items: T[]): T => items[Math.floor(random() * items.length)]!;
  for (const language of languages) {
    const isLocale = NL_LOCALES.includes(language);
    const tasks = isLocale ? matrix.nlTasks : matrix.codeTasks;
    const topics = isLocale ? matrix.nlTopics : matrix.codeTopics;
    for (const roster of matrix.roster) {
      for (const task of tasks) {
        for (const size of matrix.sizeTargets) {
          const wrapping = isLocale ? { label: 'prose', instruction: '' } : pick(matrix.wrappingStyles);
          const topic = pick(topics);
          const prompt = task.template
            .replaceAll('{language}', language)
            .replaceAll('{locale}', LOCALE_NAMES[language] ?? language)
            .replaceAll('{topic}', topic)
            .replaceAll('{size}', size.instruction)
            .replaceAll('{wrapping}', wrapping.instruction);
          cells.push({
            language,
            agent: roster.agent,
            model: roster.model,
            taskId: task.id,
            sizeLabel: size.label,
            wrappingLabel: wrapping.label,
            prompt,
          });
        }
      }
    }
  }
  return cells;
}

function plausiblyValid(output: string, cell: Cell): boolean {
  if (output.trim().length < 100) return false;
  if (cell.language === 'ja-JP' && !/[\u{3040}-\u{30FF}\u{4E00}-\u{9FFF}]/u.test(output)) return false;
  if ((cell.language === 'zh-CN' || cell.language === 'zh-TW') && !/[\u{4E00}-\u{9FFF}]/u.test(output)) return false;
  return true;
}

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const limitIndex = args.indexOf('--limit');
const limit = limitIndex !== -1 ? Number(args[limitIndex + 1]) : Number.POSITIVE_INFINITY;
const requested = args.filter((a, i) => !a.startsWith('--') && i !== limitIndex + 1);
const languages = requested.length > 0 ? requested : [...CODE_LANGUAGES, ...NL_LOCALES];

const cells = buildCells(languages).slice(0, limit);
console.log(`${cells.length} cell(s) to generate${dryRun ? ' (dry run)' : ''}`);

const seen = new Set<string>();
let index = 0;
for (const cell of cells) {
  const label = `${cell.language}/${cell.agent}:${cell.model}/${cell.taskId}/${cell.sizeLabel}`;
  if (dryRun) {
    console.log(`[dry] ${label}\n      ${cell.prompt.slice(0, 120)}...`);
    continue;
  }
  console.log(`generating ${label} ...`);
  const result = spawnSync(
    'bunx',
    [
      `@willbooster-private/agentic-workflows@${AGENTIC_WORKFLOWS_VERSION}`,
      'skills',
      'agent',
      '--agent',
      cell.agent,
      '--model',
      cell.model,
      '--cwd',
      '.',
      cell.prompt,
    ],
    { encoding: 'utf8', timeout: 2 * 60 * 60 * 1000 } // LLM responses can take up to ~2 hours.
  );
  const output = result.stdout?.trim() ?? '';
  if (result.status !== 0 || !plausiblyValid(output, cell)) {
    console.error(`  rejected (${result.status}): ${result.stderr?.slice(0, 200) ?? 'invalid output'}`);
    continue;
  }
  const fingerprint = output.slice(0, 2000);
  if (seen.has(fingerprint)) {
    console.error('  rejected: duplicate');
    continue;
  }
  seen.add(fingerprint);
  const name = `${String(index++).padStart(5, '0')}-${cell.agent}-${cell.taskId}.txt`;
  writeSample(cell.language, 'llm', name, output);
  appendManifest(cell.language, {
    file: `llm/${name}`,
    lang: cell.language,
    origin: 'llm',
    source: `${cell.agent}:${cell.model} task=${cell.taskId} size=${cell.sizeLabel} wrap=${cell.wrappingLabel}`,
    license: 'Generated (no copyright asserted)',
    sizeBucket: sizeBucketOf(Buffer.byteLength(output)),
    trainable: true,
  });
  console.log(`  saved ${name} (${output.length} chars)`);
}
