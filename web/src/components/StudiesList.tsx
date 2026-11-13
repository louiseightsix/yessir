// Studies landing page: status filter chips and the In progress / Concluded /
// Draft sections. Summaries arrive already prepared by the Astro page.
import { useState } from "react";
import { KindBadge, StatusBadge } from "./badges";
import { FilterChips } from "./controls";

/** One bar per run in study order. */
export interface RunBar {
  /** Headline metric, or null while running / flawed / missing. */
  value: number | null;
  state: "evaluated" | "in_progress" | "flawed" | "missing";
  title: string;
}

export interface StudySummary {
  slug: string;
  href: string;
  title: string;
  status: string;
  kind: string | null;
  question: string | null;
  conclusion: string | null;
  bars: RunBar[];
  /** Best headline value over valid runs, or null. */
  best: number | null;
  updated: string;
}

interface Props {
  studies: StudySummary[];
  headlineMetric: string | null;
  experiment: string;
}

const SECTIONS = [
  { status: "in_progress", label: "In progress" },
  { status: "concluded", label: "Concluded" },
  { status: "draft", label: "Draft" },
] as const;

const BAR_MAX = 28;

function barStyle(bar: RunBar): { height: number; background: string } {
  if (bar.state === "in_progress") return { height: BAR_MAX, background: "var(--c-border)" };
  if (bar.value === null) return { height: 3, background: "var(--c-border-strong)" };
  const v = Math.max(0, Math.min(1, bar.value));
  return {
    height: Math.round(3 + v * (BAR_MAX - 3)),
    background: v >= 0.95 ? "var(--c-accent)" : "var(--c-accent-soft)",
  };
}

function StudyRow({ s }: { s: StudySummary }) {
  return (
    <div className="study-row">
      <div className="study-badges">
        <StatusBadge status={s.status} />
        {s.kind && <KindBadge kind={s.kind} />}
      </div>
      <div className="study-main">
        <div className="study-title-line">
          <a className="study-title" href={s.href}>
            {s.title}
          </a>
          <span className="study-slug">{s.slug}</span>
        </div>
        {s.question && <p className="study-question">{s.question}</p>}
        {s.conclusion && (
          <p className="study-conclusion">
            <strong>Conclusion.</strong> {s.conclusion}
          </p>
        )}
      </div>
      <div className="study-bars-cell">
        {s.bars.length > 0 ? (
          <>
            <div className="study-bars">
              {s.bars.map((b, i) => (
                <span key={i} title={b.title} style={barStyle(b)} />
              ))}
            </div>
            <span className="muted small-text">
              {s.bars.length} {s.bars.length === 1 ? "run" : "runs"} · best{" "}
              <span className="mono study-best">{s.best !== null ? s.best.toFixed(2) : "—"}</span>
            </span>
          </>
        ) : (
          <span className="muted small-text">No runs yet</span>
        )}
      </div>
      <span className="study-updated">{s.updated}</span>
    </div>
  );
}

export default function StudiesList({ studies, headlineMetric, experiment }: Props) {
  type Filter = "all" | (typeof SECTIONS)[number]["status"];
  const [filter, setFilter] = useState<Filter>("all");

  if (studies.length === 0) {
    return (
      <div className="panel empty-panel">
        <p>No studies yet.</p>
        <p className="muted">
          Add one as <code>studies/{experiment}/&lt;slug&gt;.json</code> to group runs around a question.
        </p>
      </div>
    );
  }

  const count = (status: string) => studies.filter((s) => s.status === status).length;
  const options = [
    { value: "all" as const, label: "All", count: studies.length },
    ...SECTIONS.map((sec) => ({ value: sec.status, label: sec.label, count: count(sec.status) })),
  ];
  // Studies with a status outside the three known ones still get a section.
  const otherStatuses = [...new Set(studies.map((s) => s.status))].filter(
    (st) => !SECTIONS.some((sec) => sec.status === st),
  );
  const sections = [
    ...SECTIONS,
    ...otherStatuses.map((st) => ({ status: st, label: st.replaceAll("_", " ") })),
  ].filter((sec) => filter === "all" || sec.status === filter);

  return (
    <>
      <div className="studies-toolbar">
        <FilterChips label="Filter by status" options={options} selected={filter} onSelect={setFilter} />
        {headlineMetric && (
          <div className="bars-legend">
            <span className="bars-legend-key" aria-hidden="true">
              <span style={{ height: 14, background: "var(--c-accent)" }} />
              <span style={{ height: 8, background: "var(--c-accent-soft)" }} />
              <span style={{ height: 14, background: "var(--c-border)" }} />
            </span>
            {headlineMetric} per run · dark ≥ 0.95 · grey = still running
          </div>
        )}
      </div>

      {sections.map((sec) => {
        const items = studies.filter((s) => s.status === sec.status);
        if (items.length === 0 && filter === "all") return null;
        return (
          <section key={sec.status} aria-label={sec.label} className="study-section">
            <h2 className="eyebrow">
              {sec.label} · {items.length}
            </h2>
            {items.length === 0 ? (
              <p className="muted small-text">No {sec.label.toLowerCase()} studies.</p>
            ) : (
              <div className="panel table-scroll">
                <div className="study-rows">
                  {items.map((s) => (
                    <StudyRow key={s.slug} s={s} />
                  ))}
                </div>
              </div>
            )}
          </section>
        );
      })}
    </>
  );
}
