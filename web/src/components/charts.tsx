// Numbers and SVG charts: MetricCell, Sparkline, LossChart, EvalChart, SweepChart.
import { type KeyboardEvent, type PointerEvent, useState } from "react";
import {
  argMin,
  downsample,
  linearTicks,
  logTicks,
  niceStep,
  safeLog10,
  type Point,
} from "../lib/chart";
import { formatLoss, formatMetric } from "../lib/format";
import { topLevelSplits } from "../lib/splits";
import { SegmentedControl } from "./controls";

// ---------------------------------------------------------------------------
// MetricCell
// ---------------------------------------------------------------------------

/**
 * A metric value with 4 decimals and, for 0–1 metrics, a thin bar below it.
 * `muted` greys it out (flawed or not evaluated); `best` marks the best valid value in a column.
 */
export function MetricCell({
  value,
  bar = false,
  muted = false,
  best = false,
}: {
  value: number | null | undefined;
  bar?: boolean;
  muted?: boolean;
  best?: boolean;
}) {
  const hasValue = typeof value === "number";
  const showBar = bar && hasValue && value >= 0 && value <= 1;
  const cls = ["metric-cell", muted || !hasValue ? "is-muted" : "", best && !muted ? "is-best" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <span className={cls}>
      {formatMetric(value)}
      {showBar && (
        <span className="metric-bar" aria-hidden="true">
          <span style={{ width: `${(value * 100).toFixed(1)}%` }} />
        </span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

const SPARK_W = 84;
const SPARK_H = 24;

/**
 * Loss on a log scale in 84×24. The x axis spans `totalSteps`, so an in-progress
 * run only draws as far as its current step. Pass a shared `domain` to make rows
 * comparable; by default it uses the series' own min and max.
 */
export function Sparkline({
  points,
  totalSteps,
  muted = false,
  domain,
}: {
  points: Point[];
  totalSteps?: number | null;
  muted?: boolean;
  domain?: [number, number];
}) {
  if (points.length === 0) return <svg width={SPARK_W} height={SPARK_H} aria-hidden="true" />;
  const shown = downsample(points, 60);
  const xMax = Math.max(totalSteps ?? 0, points[points.length - 1].step, 1);
  const [min, max] = domain ?? [
    Math.min(...points.map((p) => p.loss)),
    Math.max(...points.map((p) => p.loss)),
  ];
  const lo = safeLog10(min);
  const hi = Math.max(safeLog10(max), lo + 1e-9);
  // Points outside a shared domain (e.g. a diverged flawed run) are pinned to the edge.
  const clampY = (y: number) => Math.min(SPARK_H - 1, Math.max(1, y));
  const coords = shown.map((p) => ({
    x: round1(1 + (p.step / xMax) * (SPARK_W - 2)),
    y: round1(clampY(SPARK_H - 2 - ((safeLog10(p.loss) - lo) / (hi - lo)) * (SPARK_H - 4))),
  }));
  return (
    <svg
      width={SPARK_W}
      height={SPARK_H}
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      aria-hidden="true"
      className="sparkline"
    >
      {shown.length === 1 ? (
        // A line needs two points; a run with one logged step gets a dot.
        <circle cx={coords[0].x} cy={coords[0].y} r={1.8} fill={muted ? "#C4C4C0" : "var(--c-accent)"} />
      ) : (
        <polyline
          points={coords.map((c) => `${c.x},${c.y}`).join(" ")}
          fill="none"
          stroke={muted ? "#C4C4C0" : "var(--c-accent)"}
          strokeWidth={1.4}
          strokeLinejoin="round"
        />
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// LossChart
// ---------------------------------------------------------------------------

export interface LossSeries {
  label: string;
  colour: string;
  points: Point[];
}

type Scale = "log" | "linear";

/**
 * Round SVG coordinates to 0.1. Node (server render) and the browser can differ
 * in the last digits of Math.log10, and React warns if the server HTML and the
 * browser's first render don't match exactly.
 */
const round1 = (v: number) => Math.round(v * 10) / 10;

/**
 * Training loss against step, with a log/linear toggle (log by default).
 * One series gets its minimum marked; several get a legend.
 */
export function LossChart({
  series,
  totalSteps,
  title,
  titleNote,
  width = 720,
  height = 250,
  markMin,
}: {
  series: LossSeries[];
  /** x axis runs to this, so an in-progress run stops part-way across. */
  totalSteps?: number | null;
  /** Heading left of the toggle, e.g. "Training loss". Plain strings, because
   *  Astro pages can't pass JSX into a client-side island. */
  title?: string;
  /** Muted mono note after the title, e.g. "train.jsonl". */
  titleNote?: string;
  width?: number;
  height?: number;
  /** Mark the minimum point. Defaults to on for a single series. */
  markMin?: boolean;
}) {
  const [scale, setScale] = useState<Scale>("log");
  const withPoints = series.filter((s) => s.points.length > 0);
  const header = (
    <div className="chart-header">
      <h2 className="chart-title">
        {title}
        {titleNote && <span className="chart-title-note"> {titleNote}</span>}
      </h2>
      {withPoints.length > 0 && (
        <SegmentedControl
          label="Y axis scale"
          size="small"
          options={[
            { value: "log", label: "Log" },
            { value: "linear", label: "Linear" },
          ]}
          selected={scale}
          onSelect={setScale}
        />
      )}
    </div>
  );

  if (withPoints.length === 0) {
    return (
      <div className="loss-chart">
        {header}
        <p className="muted">No training steps recorded.</p>
      </div>
    );
  }

  // Plot area inside the SVG's viewBox.
  const left = 56;
  const right = width - 16;
  const top = 16;
  const bottom = height - 34;

  // x: 0 → total steps (or the furthest step seen).
  const lastStep = Math.max(...withPoints.map((s) => s.points[s.points.length - 1].step));
  const xMax = Math.max(totalSteps ?? 0, lastStep, 1);
  const xOf = (step: number) => round1(left + (step / xMax) * (right - left));
  const xTicks = linearTicks(xMax, 4);

  // y: log → whole decades around the data; linear → 0 up to a round number.
  const allLoss = withPoints.flatMap((s) => s.points.map((p) => p.loss));
  const minLoss = Math.min(...allLoss);
  const maxLoss = Math.max(...allLoss);
  let yTicks: number[];
  let yOf: (v: number) => number;
  if (scale === "log") {
    // Pad the data range so the curve doesn't touch the edges; extra room at
    // the bottom leaves space for the "min" label under the lowest point.
    const span = Math.max(safeLog10(maxLoss) - safeLog10(minLoss), 0.1);
    const lo = safeLog10(minLoss) - 0.15 * span;
    const hi = safeLog10(maxLoss) + 0.04 * span;
    yTicks = logTicks(10 ** lo, 10 ** hi);
    yOf = (v) => round1(bottom - ((safeLog10(v) - lo) / (hi - lo)) * (bottom - top));
  } else {
    const step = niceStep(maxLoss, 4);
    const yMax = Math.ceil(maxLoss / step) * step;
    yTicks = linearTicks(yMax, 4);
    yOf = (v) => round1(bottom - (v / yMax) * (bottom - top));
  }

  const showMin = markMin ?? withPoints.length === 1;
  const first = withPoints[0];
  const minIdx = argMin(first.points);
  const minPoint = first.points[minIdx];

  const ariaLabel =
    withPoints.length === 1
      ? `Training loss goes from ${formatLoss(first.points[0].loss)} to ${formatLoss(
          first.points[first.points.length - 1].loss,
        )} over ${lastStep} steps; minimum ${formatLoss(minPoint.loss)} at step ${minPoint.step}`
      : `Training loss for ${withPoints.length} runs: ${withPoints
          .map((s) => `${s.label} ends at ${formatLoss(s.points[s.points.length - 1].loss)}`)
          .join(", ")}`;

  // The curve never goes below its minimum, so the label sits under the dot
  // when there's room above the axis, otherwise above it. It goes left of the
  // dot unless the dot is near the left edge.
  const minX = xOf(minPoint.step);
  const minY = yOf(minPoint.loss);
  const labelLeftOfDot = minX - left > 170;
  const labelY = minY + 18 <= bottom - 2 ? minY + 18 : minY - 10;

  return (
    <div className="loss-chart">
      {header}
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={left} x2={right} y1={yOf(t)} y2={yOf(t)} stroke="#EEEEEA" />
            <text x={left - 8} y={yOf(t) + 4} textAnchor="end" className="chart-tick">
              {t}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={xOf(t)} x2={xOf(t)} y1={bottom} y2={bottom + 5} stroke="#C4C4C0" />
            <text x={xOf(t)} y={bottom + 18} textAnchor="middle" className="chart-tick">
              {t}
            </text>
          </g>
        ))}
        <line x1={left} x2={right} y1={bottom} y2={bottom} stroke="#C4C4C0" />
        {withPoints.map((s, i) =>
          // A line needs two points; a run with one logged step gets a dot.
          s.points.length === 1 ? (
            <circle key={i} cx={xOf(s.points[0].step)} cy={yOf(s.points[0].loss)} r={3} fill={s.colour} />
          ) : (
            <polyline
              key={i}
              points={downsample(s.points, 500)
                .map((p) => `${xOf(p.step)},${yOf(p.loss)}`)
                .join(" ")}
              fill="none"
              stroke={s.colour}
              strokeWidth={1.6}
              strokeLinejoin="round"
            />
          ),
        )}
        {showMin && (
          <>
            <circle cx={minX} cy={minY} r={4} fill="#FFFFFF" stroke="#18181B" strokeWidth={1.5} />
            <text
              x={labelLeftOfDot ? minX - 6 : minX + 6}
              y={labelY}
              textAnchor={labelLeftOfDot ? "end" : "start"}
              className="chart-tick chart-ink chart-halo"
            >
              min {formatLoss(minPoint.loss)} @ {minPoint.step}
            </text>
          </>
        )}
        <text x={(left + right) / 2} y={height - 1} textAnchor="middle" className="chart-axis-label">
          step
        </text>
      </svg>
      {withPoints.length > 1 && (
        <div className="chart-legend">
          {withPoints.map((s, i) => (
            <span key={i}>
              <span className="legend-swatch" style={{ background: s.colour }} aria-hidden="true" />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvalChart
// ---------------------------------------------------------------------------

/** One `eval.jsonl` checkpoint, as passed from the page (plain data only). */
export interface EvalCheckpoint {
  step: number;
  splits: Record<string, Record<string, number | null>>;
}

/** Number of `--c-series-N` slots in tokens.css. Extra splits are not drawn. */
const SERIES_SLOTS = 8;

/** Names in first-seen order across a list of objects. */
function unionKeys(objects: Array<Record<string, unknown> | undefined>): string[] {
  const seen = new Set<string>();
  for (const obj of objects) for (const key of Object.keys(obj ?? {})) seen.add(key);
  return [...seen];
}

/**
 * One eval metric against step, one line per top-level split, from `eval.jsonl`.
 * A select picks the metric (`exact_match` by default). The y axis is fixed at
 * 0–1 because every eval metric is a fraction. Lines are labelled at their ends,
 * and the readout below lists every split's value at the hovered checkpoint (the
 * latest one otherwise), which doubles as the legend and as the text fallback for
 * the low-contrast colour slots.
 */
export function EvalChart({
  checkpoints,
  totalSteps,
  title,
  titleNote,
  width = 720,
  height = 250,
}: {
  checkpoints: EvalCheckpoint[];
  /** x axis runs to this, so an in-progress run stops part-way across. */
  totalSteps?: number | null;
  title?: string;
  titleNote?: string;
  width?: number;
  height?: number;
}) {
  const splitNames = topLevelSplits(unionKeys(checkpoints.map((c) => c.splits)));
  const metricNames = unionKeys(checkpoints.flatMap((c) => splitNames.map((s) => c.splits[s])));
  const [metric, setMetric] = useState(
    metricNames.includes("exact_match") ? "exact_match" : (metricNames[0] ?? ""),
  );
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const header = (
    <div className="chart-header">
      <h2 className="chart-title">
        {title}
        {titleNote && <span className="chart-title-note"> {titleNote}</span>}
      </h2>
      {metricNames.length > 0 && (
        <label className="chart-select">
          <span>metric</span>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {metricNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );

  if (checkpoints.length === 0 || splitNames.length === 0) {
    return (
      <div className="loss-chart">
        {header}
        <p className="muted">No evaluation checkpoints recorded.</p>
      </div>
    );
  }

  const drawn = splitNames.slice(0, SERIES_SLOTS);
  const notDrawn = splitNames.length - drawn.length;
  const series = drawn.map((name, i) => ({
    name,
    colour: `var(--c-series-${i + 1})`,
    points: checkpoints.flatMap((c) => {
      const v = c.splits[name]?.[metric];
      return typeof v === "number" ? [{ step: c.step, value: Math.min(1, Math.max(0, v)) }] : [];
    }),
  }));

  // Plot area; the right margin holds the end-of-line labels.
  const left = 48;
  const right = width - 120;
  const top = 12;
  const bottom = height - 34;
  const lastStep = checkpoints[checkpoints.length - 1].step;
  const xMax = Math.max(totalSteps ?? 0, lastStep, 1);
  const xOf = (step: number) => round1(left + (step / xMax) * (right - left));
  const yOf = (v: number) => round1(bottom - v * (bottom - top));
  const xTicks = linearTicks(xMax, 4);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  // End labels: at each line's last value, pushed apart so they never overlap.
  const LABEL_GAP = 13;
  const labels = series
    .filter((s) => s.points.length > 0)
    .map((s) => ({ name: s.name, colour: s.colour, y: yOf(s.points[s.points.length - 1].value) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP);
  }
  // If the stack runs off the bottom, shift it up as a whole.
  const overflow = labels.length > 0 ? labels[labels.length - 1].y - (bottom + 4) : 0;
  if (overflow > 0) for (const l of labels) l.y -= overflow;

  const activeIdx = hoverIdx ?? checkpoints.length - 1;
  const active = checkpoints[activeIdx];
  const valueAt = (name: string) => active.splits[name]?.[metric];

  // Pointer x (in viewBox units) → nearest checkpoint.
  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    let best = 0;
    for (let i = 1; i < checkpoints.length; i++) {
      if (Math.abs(xOf(checkpoints[i].step) - px) < Math.abs(xOf(checkpoints[best].step) - px)) best = i;
    }
    setHoverIdx(best);
  };
  const onKeyDown = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const delta = e.key === "ArrowLeft" ? -1 : 1;
    setHoverIdx(Math.min(checkpoints.length - 1, Math.max(0, activeIdx + delta)));
  };

  const ariaLabel = `${metric} by split at ${checkpoints.length} checkpoints up to step ${lastStep}; at step ${
    active.step
  }: ${drawn.map((name) => `${name} ${formatMetric(valueAt(name))}`).join(", ")}. Use left and right arrow keys to move between checkpoints.`;

  return (
    <div className="loss-chart">
      {header}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHoverIdx(null)}
        onKeyDown={onKeyDown}
        onBlur={() => setHoverIdx(null)}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={left} x2={right} y1={yOf(t)} y2={yOf(t)} stroke="#EEEEEA" />
            <text x={left - 8} y={yOf(t) + 4} textAnchor="end" className="chart-tick">
              {t}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={xOf(t)} x2={xOf(t)} y1={bottom} y2={bottom + 5} stroke="#C4C4C0" />
            <text x={xOf(t)} y={bottom + 18} textAnchor="middle" className="chart-tick">
              {t}
            </text>
          </g>
        ))}
        <line x1={left} x2={right} y1={bottom} y2={bottom} stroke="#C4C4C0" />
        {hoverIdx !== null && (
          <line x1={xOf(active.step)} x2={xOf(active.step)} y1={top} y2={bottom} stroke="#C4C4C0" />
        )}
        {series.map((s) =>
          s.points.length > 1 ? (
            <polyline
              key={`l${s.name}`}
              points={s.points.map((p) => `${xOf(p.step)},${yOf(p.value)}`).join(" ")}
              fill="none"
              stroke={s.colour}
              strokeWidth={2}
              strokeLinejoin="round"
            />
          ) : null,
        )}
        {series.map((s) =>
          s.points.map((p) => (
            // A 2px panel-coloured ring keeps overlapping markers apart.
            <circle
              key={`m${s.name}${p.step}`}
              cx={xOf(p.step)}
              cy={yOf(p.value)}
              r={4}
              fill={s.colour}
              stroke="var(--c-panel)"
              strokeWidth={2}
            />
          )),
        )}
        {labels.map((l) => (
          <g key={`t${l.name}`}>
            <line x1={right + 6} x2={right + 18} y1={l.y} y2={l.y} stroke={l.colour} strokeWidth={2} />
            <text x={right + 22} y={l.y + 4} className="chart-tick chart-ink">
              {l.name}
            </text>
          </g>
        ))}
        <text x={(left + right) / 2} y={height - 1} textAnchor="middle" className="chart-axis-label">
          step
        </text>
      </svg>
      <div className="chart-legend eval-readout" aria-live="polite">
        <span className="eval-readout-step">step {active.step}</span>
        {series.map((s) => (
          <span key={s.name}>
            <span className="legend-swatch" style={{ background: s.colour }} aria-hidden="true" />
            {s.name} <strong>{formatMetric(valueAt(s.name))}</strong>
          </span>
        ))}
        {notDrawn > 0 && <span className="muted">+{notDrawn} more splits not drawn</span>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SweepChart
// ---------------------------------------------------------------------------

export interface SweepPoint {
  /** The swept setting's value, as shown on the x axis. */
  x: string;
  value: number;
}

/**
 * Headline metric against the swept setting: a line with dots and value labels.
 * x positions are evenly spaced categories in the order given (the caller sorts),
 * like the mock's 1, 2, 4, 6, 8. Metrics in 0–1 get a fixed 0–1 axis.
 */
export function SweepChart({
  points,
  metric,
  setting,
  width = 460,
  height = 240,
}: {
  points: SweepPoint[];
  metric: string;
  setting: string;
  width?: number;
  height?: number;
}) {
  const left = 44;
  const right = width - 20;
  const top = 28; // room for the value label above a point at the top
  const bottom = height - 40;

  const categories = [...new Set(points.map((p) => p.x))];
  const xOf = (x: string) => {
    const i = categories.indexOf(x);
    return round1(categories.length === 1 ? (left + right) / 2 : left + (i / (categories.length - 1)) * (right - left));
  };

  const values = points.map((p) => p.value);
  const unit = values.every((v) => v >= 0 && v <= 1);
  const maxValue = Math.max(...values, 1e-9);
  const yStep = niceStep(maxValue, 2);
  const yMax = unit ? 1 : Math.ceil(maxValue / yStep) * yStep;
  const yTicks = unit ? [0, 0.5, 1] : linearTicks(yMax, 2);
  const yOf = (v: number) => round1(bottom - (v / yMax) * (bottom - top));

  const line = points.map((p) => `${xOf(p.x)},${yOf(p.value)}`).join(" ");
  const ariaLabel = `${metric} by ${setting}: ${points.map((p) => `${p.x} → ${p.value.toFixed(2)}`).join(", ")}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel} className="sweep-chart">
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={left} x2={right} y1={yOf(t)} y2={yOf(t)} stroke="#EEEEEA" />
          <text x={left - 8} y={yOf(t) + 4} textAnchor="end" className="chart-tick">
            {unit ? t.toFixed(1) : t}
          </text>
        </g>
      ))}
      {categories.map((c) => (
        <text key={`x${c}`} x={xOf(c)} y={bottom + 22} textAnchor="middle" className="chart-tick">
          {c}
        </text>
      ))}
      <polyline points={line} fill="none" stroke="var(--c-accent)" strokeWidth={2} />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={xOf(p.x)} cy={yOf(p.value)} r={4.5} fill="#FFFFFF" stroke="var(--c-accent)" strokeWidth={2} />
          <text x={xOf(p.x)} y={yOf(p.value) - 10} textAnchor="middle" className="chart-tick chart-ink chart-halo">
            {p.value.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={(left + right) / 2} y={height - 2} textAnchor="middle" className="chart-axis-label">
        {setting}
      </text>
    </svg>
  );
}
