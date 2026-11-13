// The Runs table: search, status/variant/study filters, row selection with a
// compare bar, and "Load older runs" paging. Rows arrive already prepared by
// the Astro page, so this component only filters and draws them.
import { Fragment, useState } from "react";
import type { Point } from "../lib/chart";
import type { ChangedSetting, RunStatus } from "../lib/data";
import { formatDuration, formatLoss, formatValue } from "../lib/format";
import { DiffChips, DirtyTag, FlawedTag, OutdatedTag, RunName } from "./badges";
import { MetricCell, Sparkline } from "./charts";
import { FilterChips, ProgressBar } from "./controls";

export interface RunRow {
  name: string;
  href: string;
  variant: string | null;
  status: RunStatus;
  flawedReason: string | null;
  outdatedReason: string | null;
  dirty: boolean;
  changed: ChangedSetting[];
  studies: Array<{ slug: string; href: string }>;
  metrics: Record<string, number> | null;
  /** Downsampled loss curve for the sparkline. */
  spark: Point[];
  lastStep: number | null;
  lastLoss: number | null;
  totalSteps: number | null;
  elapsedSec: number | null;
  /** README text, searched along with the name and settings. */
  notes: string;
}

interface Props {
  rows: RunRow[];
  metricNames: string[];
  /** Gets the 0–1 bar, like the mock's exact_match column. */
  headlineMetric: string | null;
  variants: string[];
  studySlugs: string[];
}

type StatusFilter = "all" | RunStatus;
const PAGE_SIZE = 50;
/** Study filter values besides a slug. */
const ANY_STUDY = "__any__";
const NO_STUDY = "__none__";

function matchesSearch(row: RunRow, query: string): boolean {
  if (!query) return true;
  const haystack = [
    row.name,
    row.notes,
    row.flawedReason ?? "",
    row.outdatedReason ?? "",
    ...row.changed.map((c) => `${c.name}=${formatValue(c.value)}`),
  ]
    .join("\n")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

export default function RunsTable({ rows, metricNames, headlineMetric, variants, studySlugs }: Props) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [variant, setVariant] = useState("all");
  const [study, setStudy] = useState(ANY_STUDY);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [shownCount, setShownCount] = useState(PAGE_SIZE);

  // Status chip counts ignore the status filter itself, but respect the others.
  const otherFiltersMatch = rows.filter(
    (r) =>
      matchesSearch(r, query) &&
      (variant === "all" || r.variant === variant) &&
      (study === ANY_STUDY ||
        (study === NO_STUDY ? r.studies.length === 0 : r.studies.some((s) => s.slug === study))),
  );
  const filtered = otherFiltersMatch.filter((r) => status === "all" || r.status === status);
  const shown = filtered.slice(0, shownCount);

  const countOf = (s: RunStatus) => otherFiltersMatch.filter((r) => r.status === s).length;
  const statusOptions = [
    { value: "all" as const, label: "All", count: otherFiltersMatch.length },
    { value: "evaluated" as const, label: "Evaluated", count: countOf("evaluated") },
    { value: "in_progress" as const, label: "Running", count: countOf("in_progress") },
    { value: "flawed" as const, label: "Flawed", count: countOf("flawed") },
  ];

  // One log-scale domain for every sparkline, so their heights are comparable.
  // Flawed runs are left out (unless every run is flawed) so a diverged one doesn't flatten the rest.
  const notFlawed = rows.filter((r) => r.status !== "flawed");
  const allLoss = (notFlawed.length > 0 ? notFlawed : rows).flatMap((r) => r.spark.map((p) => p.loss));
  const sparkDomain: [number, number] | undefined =
    allLoss.length > 0 ? [Math.min(...allLoss), Math.max(...allLoss)] : undefined;

  function toggle(name: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  return (
    <>
      <div className="filter-bar">
        <label className="search-box">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
            <circle cx="7" cy="7" r="4.5" />
            <path d="M10.5 10.5 14 14" />
          </svg>
          <span className="visually-hidden">Search runs</span>
          <input
            type="search"
            placeholder="Search name, setting, note…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <FilterChips label="Filter by status" options={statusOptions} selected={status} onSelect={setStatus} />
        <label className="inline-select">
          Variant
          <select value={variant} onChange={(e) => setVariant(e.target.value)}>
            <option value="all">all</option>
            {variants.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="inline-select">
          Study
          <select value={study} onChange={(e) => setStudy(e.target.value)}>
            <option value={ANY_STUDY}>any</option>
            {studySlugs.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
            <option value={NO_STUDY}>none</option>
          </select>
        </label>
      </div>

      {selected.size >= 2 && (
        <div className="compare-bar">
          <span className="compare-count">{selected.size} runs selected</span>
          <span className="compare-hint">
            Compare overlays loss curves and lines up settings and metrics side by side.
          </span>
          <span className="spacer" />
          <button type="button" className="compare-clear" onClick={() => setSelected(new Set())}>
            Clear
          </button>
          <button type="button" className="compare-go" disabled title="The compare view isn't designed yet">
            Compare →
          </button>
        </div>
      )}

      <div className="panel table-scroll">
        <table className="data-table runs-table">
          <thead>
            <tr>
              <th scope="col" className="col-check">
                <span className="visually-hidden">Select</span>
              </th>
              <th scope="col">
                Run <span className="th-note">↓ newest</span>
              </th>
              <th scope="col">Variant</th>
              <th scope="col">Changed from default</th>
              <th scope="col">Studies</th>
              {metricNames.map((m) => (
                <th key={m} scope="col" className="num metric-th">
                  {/* Allow a line break after each "_", so a long metric name
                      doesn't make its column wider than its numbers need. */}
                  {m.split("_").map((part, i) => (
                    <Fragment key={i}>
                      {i > 0 && (
                        <>
                          _<wbr />
                        </>
                      )}
                      {part}
                    </Fragment>
                  ))}
                </th>
              ))}
              <th scope="col">Loss</th>
              <th scope="col" className="num">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 && (
              <tr>
                <td colSpan={7 + metricNames.length} className="empty-row">
                  {rows.length === 0 ? "No runs yet." : "No runs match these filters."}
                </td>
              </tr>
            )}
            {shown.map((r) => {
              const isSelected = selected.has(r.name);
              const rowClass = r.status === "flawed" ? "row-flawed" : isSelected ? "row-selected" : undefined;
              const running = r.status === "in_progress";
              return (
                <tr key={r.name} className={rowClass}>
                  <td className="col-check">
                    <label className="check-hit">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggle(r.name)}
                        aria-label={`Select ${r.name}`}
                      />
                    </label>
                  </td>
                  <td>
                    <RunName name={r.name} href={r.href} />
                    {(running || r.dirty || r.flawedReason !== null || r.outdatedReason !== null) && (
                      <div className="run-tags">
                        {running && (
                          <>
                            <span className="running-label">
                              <span className="running-dot" aria-hidden="true" />
                              running · {r.lastStep ?? 0}
                              {r.totalSteps !== null && ` / ${r.totalSteps}`}
                            </span>
                            {r.totalSteps !== null && (
                              <ProgressBar fraction={(r.lastStep ?? 0) / r.totalSteps} width={90} />
                            )}
                          </>
                        )}
                        {r.dirty && <DirtyTag />}
                        {r.flawedReason !== null && <FlawedTag reason={r.flawedReason} />}
                        {r.outdatedReason !== null && <OutdatedTag reason={r.outdatedReason} />}
                      </div>
                    )}
                  </td>
                  <td className="mono-cell">{r.variant ?? "—"}</td>
                  <td>
                    <div className="cell-wrap">
                      <DiffChips changed={r.changed} />
                    </div>
                  </td>
                  <td>
                    <div className="cell-wrap">
                      {r.studies.map((s) => (
                        <a key={s.slug} className="study-chip" href={s.href}>
                          <span className="mono">{s.slug}</span>
                        </a>
                      ))}
                    </div>
                  </td>
                  {metricNames.map((m) => (
                    <td key={m} className="num">
                      <MetricCell
                        value={r.metrics?.[m]}
                        bar={m === headlineMetric}
                        muted={r.status === "flawed"}
                      />
                    </td>
                  ))}
                  <td>
                    <div className="loss-cell">
                      <Sparkline
                        points={r.spark}
                        totalSteps={r.totalSteps}
                        muted={r.status === "flawed"}
                        domain={sparkDomain}
                      />
                      <span>{r.lastLoss !== null ? formatLoss(r.lastLoss) : "—"}</span>
                    </div>
                  </td>
                  <td className="num muted-cell">
                    {r.elapsedSec !== null ? formatDuration(r.elapsedSec) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        <span>
          Showing {shown.length} of {filtered.length}
        </span>
        {filtered.length > shown.length && (
          <button type="button" className="button-secondary" onClick={() => setShownCount((n) => n + PAGE_SIZE)}>
            Load older runs
          </button>
        )}
      </div>
    </>
  );
}
