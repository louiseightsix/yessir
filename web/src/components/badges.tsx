// Labels, tags and chips. Plain React with no state, so Astro pages render them
// to static HTML and the interactive islands (Runs table, builder) reuse them.
import type { ChangedSetting, RunStatus } from "../lib/data";
import { formatValue, splitRunName } from "../lib/format";

/** Study lifecycle: `draft` amber, `in_progress` blue, `concluded` green. */
export function StatusBadge({ status }: { status: string }) {
  return <span className={`pill study-${status}`}>{status.replaceAll("_", " ")}</span>;
}

/** Study design: exploratory / ablation / comparison / sweep. */
export function KindBadge({ kind }: { kind: string }) {
  return <span className="kind-badge">{kind}</span>;
}

const RUN_STATUS_LABEL: Record<RunStatus, string> = {
  evaluated: "evaluated",
  in_progress: "running",
  flawed: "flawed",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return <span className={`pill run-${status}`}>{RUN_STATUS_LABEL[status]}</span>;
}

/** One `key=value` chip per setting that differs from the variant default. */
export function DiffChips({ changed }: { changed: ChangedSetting[] }) {
  if (changed.length === 0) return <span className="muted small-text">defaults</span>;
  return (
    <span className="chip-row">
      {changed.map((c) => (
        <span key={c.name} className="diff-chip">
          {c.name}={formatValue(c.value)}
        </span>
      ))}
    </span>
  );
}

export function DirtyTag() {
  return (
    <span className="dirty-tag" title="Working tree had uncommitted changes">
      dirty
    </span>
  );
}

/** Solid FLAWED pill plus the one-line reason. */
export function FlawedTag({ reason }: { reason: string }) {
  return (
    <span className="flawed-tag">
      <span className="flawed-pill">FLAWED</span>
      {reason && <span className="flawed-reason">{reason}</span>}
    </span>
  );
}

export function FlawedBanner({ reason }: { reason: string }) {
  return (
    <div role="alert" className="flawed-banner">
      <span className="flawed-pill">FLAWED</span>
      <div>
        {reason && <strong>{reason}</strong>}
        <span>Kept for the record. Don't cite these numbers.</span>
      </div>
    </div>
  );
}

/** Grey OUTDATED pill plus the one-line reason. */
export function OutdatedTag({ reason }: { reason: string }) {
  return (
    <span className="outdated-tag">
      <span className="outdated-pill">OUTDATED</span>
      {reason && <span className="outdated-reason">{reason}</span>}
    </span>
  );
}

export function OutdatedBanner({ reason }: { reason: string }) {
  return (
    <div role="note" className="outdated-banner">
      <span className="outdated-pill">OUTDATED</span>
      <div>
        {reason && <strong>{reason}</strong>}
        <span>Sound under an earlier spec, but not a current result.</span>
      </div>
    </div>
  );
}

/** `YYYY-MM-DD-<sha>-` muted, `adjective-noun` bold. A link when `href` is given. */
export function RunName({ name, href }: { name: string; href?: string }) {
  const { prefix, short } = splitRunName(name);
  const inner = (
    <>
      <span className="run-name-prefix">{prefix}</span>
      <span className="run-name-short">{short}</span>
    </>
  );
  return href ? (
    <a className="run-name" href={href}>
      {inner}
    </a>
  ) : (
    <span className="run-name">{inner}</span>
  );
}

/** Outline pill linking to a study; with a label it reads `<slug> as <label>`. */
export function StudyChip({ slug, label, href }: { slug: string; label?: string | null; href: string }) {
  return (
    <a className="study-chip" href={href}>
      <span className="mono">{slug}</span>
      {label && (
        <>
          <span className="muted">as</span>
          <span className="study-chip-label">{label}</span>
        </>
      )}
    </a>
  );
}
