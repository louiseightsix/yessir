// Small formatting helpers shared by pages and components.

/** A config value as it appears in `config.json`: `4`, `"all"`, `null`. */
export function formatValue(value: unknown): string {
  return JSON.stringify(value) ?? String(value);
}

/** Eval metrics always show 4 decimals so columns line up. */
export function formatMetric(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(4) : "—";
}

/** Loss with more decimals once it's small: 12.18, 3.758, 0.0141. */
export function formatLoss(value: number): string {
  if (value >= 10) return value.toFixed(2);
  if (value >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

/** Seconds as "45s", "2m 04s" or "1h 05m". */
export function formatDuration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/**
 * Split a run name into the muted prefix and the memorable part:
 * "2026-11-12-a13cc88-closing-filly" → ["2026-11-12-a13cc88-", "closing-filly"].
 */
export function splitRunName(name: string): { prefix: string; short: string } {
  const match = /^(\d{4}-\d{2}-\d{2}-[0-9a-f]+-)(.+)$/.exec(name);
  return match ? { prefix: match[1], short: match[2] } : { prefix: "", short: name };
}
