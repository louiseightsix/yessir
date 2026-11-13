// Scale and tick helpers for the SVG charts. No React here, so they're easy to test by eye.

export interface Point {
  step: number;
  loss: number;
}

/** Nice tick spacing (1, 2, 2.5 or 5 × a power of ten) giving about `count` intervals. */
export function niceStep(span: number, count: number): number {
  if (span <= 0) return 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
  return nice * power;
}

/** Ticks 0, step, 2·step, … up to `max`. With max = 1000 and count 4 this is every 250. */
export function linearTicks(max: number, count = 4): number[] {
  const step = niceStep(max, count);
  const ticks: number[] = [];
  for (let v = 0; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

/**
 * Log-scale ticks inside [min, max]. Wide ranges get powers of ten (0.01, 0.1, 1, 10);
 * narrow ones (under two decades) also get 2× and 5× so the axis still has labels.
 */
export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const multipliers = hi - lo <= 2 ? [1, 2, 5] : [1];
  const ticks: number[] = [];
  for (let e = lo; e <= hi; e++) {
    for (const m of multipliers) {
      const v = Number((m * 10 ** e).toPrecision(12));
      if (v >= min && v <= max) ticks.push(v);
    }
  }
  return ticks;
}

/**
 * Keep at most `max` points, evenly spaced, always including the last one.
 * A 10 000-step run would otherwise put 10 000 vertices in every sparkline.
 */
export function downsample<T>(points: T[], max: number): T[] {
  if (points.length <= max) return points;
  const stride = (points.length - 1) / (max - 1);
  const out: T[] = [];
  for (let i = 0; i < max; i++) out.push(points[Math.round(i * stride)]);
  return out;
}

/** Index of the lowest-loss point, or -1 for an empty series. */
export function argMin(points: Point[]): number {
  let best = -1;
  for (let i = 0; i < points.length; i++) {
    if (best === -1 || points[i].loss < points[best].loss) best = i;
  }
  return best;
}

/** Loss is always > 0, but guard log10 against a stray 0. */
const LOG_FLOOR = 1e-6;
export function safeLog10(v: number): number {
  return Math.log10(Math.max(v, LOG_FLOOR));
}

/**
 * `n` colours from the sequential sweep ramp (light → dark), spread across it.
 * With more than five series, colours are interpolated between ramp steps.
 */
const RAMP = ["#A9BDEB", "#7E9BE0", "#4F75D6", "#2F5BD3", "#1B3A8F"];
export function rampColours(n: number): string[] {
  if (n <= 0) return [];
  if (n === 1) return [RAMP[3]];
  return Array.from({ length: n }, (_, i) => {
    const pos = (i / (n - 1)) * (RAMP.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, RAMP.length - 1);
    return mixHex(RAMP[lo], RAMP[hi], pos - lo);
  });
}

function mixHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return `#${pa
    .map((c, i) => Math.round(c + (pb[i] - c) * t).toString(16).padStart(2, "0"))
    .join("")}`;
}
