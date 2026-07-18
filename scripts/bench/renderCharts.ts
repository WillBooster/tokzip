/**
 * Renders the benchmark report as standalone SVG charts for the README and dashboard.
 *
 * Emits light/dark variants of two figures: a compression-ratio-vs-speed scatter
 * (Pareto view) and a per-language ratio dot plot. The SVGs are self-contained
 * (system fonts, explicit background) so they render identically on GitHub and
 * when opened directly from GitHub Pages.
 *
 * Usage: bun scripts/bench/renderCharts.ts <bench.json> <outputDir>
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface ChartReport {
  total: { docs: number; inputBytes: number; ratios: Record<string, number> };
  languages: Record<string, { docs: number; total: { ratios: Record<string, number> } }>;
  speed?: Record<string, { compressMBps: number; decompressMBps: number }>;
}

interface Theme {
  mode: 'light' | 'dark';
  /** Matches GitHub's page background so embedded figures blend in. */
  background: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  grid: string;
  axis: string;
  seriesFast: string;
  seriesSmall: string;
  seriesBrotli: string;
  seriesZstd: string;
  competitor: string;
}

interface PlacedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

const THEMES: Theme[] = [
  {
    mode: 'light',
    background: '#ffffff',
    textPrimary: '#0b0b0b',
    textSecondary: '#52514e',
    textMuted: '#898781',
    grid: '#e1e0d9',
    axis: '#c3c2b7',
    seriesFast: '#2a78d6',
    seriesSmall: '#1baf7a',
    seriesBrotli: '#eda100',
    seriesZstd: '#4a3aa7',
    competitor: '#898781',
  },
  {
    mode: 'dark',
    background: '#0d1117',
    textPrimary: '#ffffff',
    textSecondary: '#c3c2b7',
    textMuted: '#898781',
    grid: '#2c2c2a',
    axis: '#383835',
    seriesFast: '#3987e5',
    seriesSmall: '#199e70',
    seriesBrotli: '#c98500',
    seriesZstd: '#9085e9',
    competitor: '#898781',
  },
];

const FONT_FAMILY = "-apple-system, 'Segoe UI', system-ui, sans-serif";
/** Approximate glyph advance at 12.5px for collision boxes (system sans average). */
const LABEL_CHAR_WIDTH = 6.6;

main();

function main(): void {
  const [reportPath, outputDir] = process.argv.slice(2);
  if (!reportPath || !outputDir) {
    console.error('usage: bun scripts/bench/renderCharts.ts <bench.json> <outputDir>');
    process.exit(1);
  }
  const report = JSON.parse(readFileSync(reportPath, 'utf8')) as ChartReport;
  mkdirSync(outputDir, { recursive: true });
  for (const theme of THEMES) {
    writeChart(outputDir, `ratio-speed-${theme.mode}.svg`, renderRatioSpeedChart(report, theme));
    writeChart(outputDir, `languages-${theme.mode}.svg`, renderLanguageChart(report, theme));
  }
}

function writeChart(outputDir: string, fileName: string, svg: string | undefined): void {
  if (svg === undefined) return;
  const path = join(outputDir, fileName);
  writeFileSync(path, svg);
  console.log(`wrote ${path}`);
}

/** Scatter of overall compression speed (log x) vs output ratio; bottom-right is better. */
function renderRatioSpeedChart(report: ChartReport, theme: Theme): string | undefined {
  if (!report.speed) return undefined;
  const points = Object.entries(report.speed)
    .filter(([method]) => typeof report.total.ratios[method] === 'number')
    .map(([method, speed]) => ({
      method,
      speed: speed.compressMBps,
      ratio: report.total.ratios[method]! * 100,
    }));
  if (points.length === 0) return undefined;

  const width = 880;
  const height = 460;
  const plot = { left: 64, top: 76, right: width - 28, bottom: height - 56 };
  const speeds = points.map((point) => point.speed);
  const ratios = points.map((point) => point.ratio);
  const xTicks = logTicks(Math.min(...speeds), Math.max(...speeds));
  const yMin = Math.max(0, Math.floor((Math.min(...ratios) - 4) / 10) * 10);
  const yMax = Math.ceil((Math.max(...ratios) + 4) / 10) * 10;
  const x = (speed: number): number =>
    plot.left +
    ((Math.log10(speed) - Math.log10(xTicks[0]!)) / (Math.log10(xTicks.at(-1)!) - Math.log10(xTicks[0]!))) *
      (plot.right - plot.left);
  const y = (ratio: number): number => plot.bottom - ((ratio - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);

  const parts: string[] = [];
  for (let ratio = yMin; ratio <= yMax; ratio += 10) {
    parts.push(
      `<line x1="${plot.left}" y1="${y(ratio)}" x2="${plot.right}" y2="${y(ratio)}" stroke="${theme.grid}" stroke-width="1"/>`,
      `<text x="${plot.left - 8}" y="${y(ratio) + 4}" text-anchor="end" font-size="12" fill="${theme.textMuted}">${ratio}%</text>`
    );
  }
  for (const tick of xTicks) {
    parts.push(
      `<line x1="${x(tick)}" y1="${plot.top}" x2="${x(tick)}" y2="${plot.bottom}" stroke="${theme.grid}" stroke-width="1"/>`,
      `<text x="${x(tick)}" y="${plot.bottom + 18}" text-anchor="middle" font-size="12" fill="${theme.textMuted}">${tick}</text>`
    );
  }
  parts.push(
    `<line x1="${plot.left}" y1="${plot.bottom}" x2="${plot.right}" y2="${plot.bottom}" stroke="${theme.axis}" stroke-width="1"/>`
  );

  // Labels are placed greedily so per-run data movements cannot overlap them.
  const placedBoxes: PlacedBox[] = points.map((point) => ({
    x: x(point.speed) - 8,
    y: y(point.ratio) - 8,
    width: 16,
    height: 16,
  }));
  for (const point of points.toSorted((left, right) => left.speed - right.speed)) {
    const pointX = x(point.speed);
    const pointY = y(point.ratio);
    const color = seriesColor(point.method, theme);
    const label = displayName(point.method);
    const labelWidth = label.length * LABEL_CHAR_WIDTH;
    const spot = placeLabel(pointX, pointY, labelWidth, placedBoxes, plot);
    parts.push(
      `<circle cx="${pointX}" cy="${pointY}" r="6" fill="${color}" stroke="${theme.background}" stroke-width="2"/>`,
      `<text x="${spot.x}" y="${spot.y}" font-size="12.5" font-weight="${point.method.startsWith('tokzip') ? 600 : 400}" fill="${theme.textSecondary}">${escapeXml(label)}</text>`
    );
  }

  const megabytes = (report.total.inputBytes / 1_048_576).toFixed(1);
  parts.push(
    `<text x="24" y="30" font-size="15" font-weight="600" fill="${theme.textPrimary}">Compression ratio vs speed</text>`,
    `<text x="24" y="50" font-size="12" fill="${theme.textSecondary}">bench-v2 corpus (${report.total.docs} docs, ${megabytes} MB), text channel, dictionary-free secondary metric (see the dashboard for the session-amortized primary) — closer to the bottom-right is better</text>`,
    `<text x="${(plot.left + plot.right) / 2}" y="${height - 16}" text-anchor="middle" font-size="12" fill="${theme.textSecondary}">compression speed (MB/s, log scale) →</text>`,
    `<text transform="translate(18 ${(plot.top + plot.bottom) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${theme.textSecondary}">← output / input (%)</text>`
  );
  return svgDocument(width, height, theme, parts.join('\n'));
}

/** Dot plot of per-language ratios: tokzip modes vs the strongest general-purpose codecs. */
function renderLanguageChart(report: ChartReport, theme: Theme): string {
  const series = [
    { method: 'tokzip small', color: theme.seriesSmall, shape: 'circle' as const },
    { method: 'tokzip fast', color: theme.seriesFast, shape: 'triangle' as const },
    // The browser-native reference codec first, the strongest server-side codec second.
    { method: 'b64url(cs gzip)', color: theme.seriesZstd, shape: 'diamond' as const },
    { method: 'b64url(brotli q11)', color: theme.seriesBrotli, shape: 'square' as const },
    // The filter only guards the optional competitors: bench.ts unconditionally measures
    // both tokzip modes, so `series` (and the Math.max spread below) is never empty for
    // any report the benchmark actually produces.
  ].filter((entry) => Object.values(report.languages).some((language) => language.total.ratios[entry.method]));
  const languages = Object.keys(report.languages).toSorted(
    (left, right) =>
      (report.languages[left]!.total.ratios['tokzip small'] ?? 1) -
      (report.languages[right]!.total.ratios['tokzip small'] ?? 1)
  );

  const width = 880;
  const rowHeight = 26;
  const plot = { left: 110, top: 108, right: width - 28 };
  const plotBottom = plot.top + languages.length * rowHeight;
  const height = plotBottom + 52;
  const maxRatio =
    Math.ceil(
      Math.max(
        ...languages.flatMap((language) =>
          series.map((entry) => (report.languages[language]!.total.ratios[entry.method] ?? 0) * 100)
        )
      ) / 10
    ) * 10;
  const x = (ratio: number): number => plot.left + (ratio / maxRatio) * (plot.right - plot.left);

  const parts: string[] = [];
  for (let ratio = 0; ratio <= maxRatio; ratio += 10) {
    parts.push(
      `<line x1="${x(ratio)}" y1="${plot.top}" x2="${x(ratio)}" y2="${plotBottom}" stroke="${theme.grid}" stroke-width="1"/>`,
      `<text x="${x(ratio)}" y="${plotBottom + 18}" text-anchor="middle" font-size="12" fill="${theme.textMuted}">${ratio}%</text>`
    );
  }
  for (const [row, language] of languages.entries()) {
    const centerY = plot.top + row * rowHeight + rowHeight / 2;
    parts.push(
      `<line x1="${plot.left}" y1="${centerY}" x2="${plot.right}" y2="${centerY}" stroke="${theme.grid}" stroke-width="1"/>`,
      `<text x="${plot.left - 10}" y="${centerY + 4}" text-anchor="end" font-size="12.5" fill="${theme.textPrimary}">${escapeXml(language)}</text>`
    );
    for (const entry of series) {
      const ratio = report.languages[language]!.total.ratios[entry.method];
      if (typeof ratio !== 'number') continue;
      parts.push(marker(entry.shape, x(ratio * 100), centerY, entry.color, theme.background));
    }
  }

  let legendX = plot.left;
  for (const entry of series) {
    const label = displayName(entry.method);
    parts.push(
      marker(entry.shape, legendX + 6, 76, entry.color, theme.background),
      `<text x="${legendX + 17}" y="80" font-size="12.5" fill="${theme.textSecondary}">${escapeXml(label)}</text>`
    );
    legendX += 17 + label.length * LABEL_CHAR_WIDTH + 26;
  }
  parts.push(
    `<text x="24" y="30" font-size="15" font-weight="600" fill="${theme.textPrimary}">Compression ratio by language</text>`,
    `<text x="24" y="50" font-size="12" fill="${theme.textSecondary}">output / input (%) on the bench-v2 split, text channel, dictionary-free secondary metric — lower is better</text>`,
    `<text x="${(plot.left + plot.right) / 2}" y="${height - 12}" text-anchor="middle" font-size="12" fill="${theme.textSecondary}">output / input (%)</text>`
  );
  return svgDocument(width, height, theme, parts.join('\n'));
}

/**
 * Finds a label spot that avoids every occupied box, trying right, above, below,
 * then left of the mark. Falls back to the right when the plot is too crowded.
 */
function placeLabel(
  pointX: number,
  pointY: number,
  labelWidth: number,
  placedBoxes: PlacedBox[],
  plot: { left: number; top: number; right: number; bottom: number }
): { x: number; y: number } {
  const candidates = [
    { x: pointX + 10, y: pointY + 4 },
    { x: pointX - labelWidth / 2, y: pointY - 12 },
    { x: pointX - labelWidth / 2, y: pointY + 20 },
    { x: pointX - 10 - labelWidth, y: pointY + 4 },
  ];
  const fits = (box: PlacedBox): boolean =>
    box.x >= plot.left - 40 &&
    box.x + box.width <= plot.right + 24 &&
    box.y >= plot.top - 4 &&
    box.y + box.height <= plot.bottom + 4 &&
    placedBoxes.every(
      (placed) =>
        box.x + box.width < placed.x ||
        placed.x + placed.width < box.x ||
        box.y + box.height < placed.y ||
        placed.y + placed.height < box.y
    );
  for (const candidate of candidates) {
    const box = { x: candidate.x, y: candidate.y - 11, width: labelWidth, height: 14 };
    if (fits(box)) {
      placedBoxes.push(box);
      return candidate;
    }
  }
  placedBoxes.push({ x: candidates[0]!.x, y: candidates[0]!.y - 11, width: labelWidth, height: 14 });
  return candidates[0]!;
}

/** Hue is doubled with shape so series stay distinguishable under color-vision deficiency. */
function marker(
  shape: 'circle' | 'triangle' | 'square' | 'diamond',
  centerX: number,
  centerY: number,
  color: string,
  ring: string
): string {
  const stroke = `fill="${color}" stroke="${ring}" stroke-width="2"`;
  switch (shape) {
    case 'circle': {
      return `<circle cx="${centerX}" cy="${centerY}" r="5.5" ${stroke}/>`;
    }
    case 'triangle': {
      return `<path d="M${centerX} ${centerY - 6} L${centerX + 6} ${centerY + 5} L${centerX - 6} ${centerY + 5} Z" ${stroke}/>`;
    }
    case 'square': {
      return `<rect x="${centerX - 5}" y="${centerY - 5}" width="10" height="10" rx="1.5" ${stroke}/>`;
    }
    case 'diamond': {
      return `<path d="M${centerX} ${centerY - 6.5} L${centerX + 6.5} ${centerY} L${centerX} ${centerY + 6.5} L${centerX - 6.5} ${centerY} Z" ${stroke}/>`;
    }
  }
}

function seriesColor(method: string, theme: Theme): string {
  if (method === 'tokzip fast') return theme.seriesFast;
  if (method === 'tokzip small') return theme.seriesSmall;
  return theme.competitor;
}

/** Log-scale tick values (1-2-5 sequence) covering [min, max] with one step of padding. */
function logTicks(min: number, max: number): number[] {
  const steps: number[] = [];
  for (let magnitude = -2; magnitude <= 5; magnitude++) {
    for (const mantissa of [1, 2, 5]) steps.push(mantissa * 10 ** magnitude);
  }
  const first = Math.max(0, steps.findLastIndex((step) => step <= min) - 1);
  const last = Math.min(steps.length - 1, steps.findIndex((step) => step >= max) + 1);
  return steps.slice(first, last + 1).map((step) => Number(step.toPrecision(2)));
}

/** Drops the base64url framing prefix: the figure subtitle states the text-channel rule. */
function displayName(method: string): string {
  const framed = /^b64url\((.+)\)$/.exec(method);
  return (framed ? framed[1]! : method).replace(' URI', '');
}

function svgDocument(width: number, height: number, theme: Theme, body: string): string {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" ` +
    `font-family="${FONT_FAMILY}">\n<rect width="${width}" height="${height}" fill="${theme.background}"/>\n${body}\n</svg>\n`
  );
}

function escapeXml(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}
