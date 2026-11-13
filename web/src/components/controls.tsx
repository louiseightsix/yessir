// Controls with no state of their own: the parent holds the selection and passes
// `onSelect`. Toggle groups use `aria-pressed` so screen readers announce the choice.

export function ProgressBar({ fraction, width }: { fraction: number; width?: number }) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return (
    <span className="progress-track" style={width ? { width } : undefined}>
      <span className="progress-fill" style={{ width: `${pct.toFixed(1)}%` }} />
    </span>
  );
}

export interface ChipOption<T extends string> {
  value: T;
  label: string;
  count?: number;
}

/** Pill buttons; the selected one is solid ink. */
export function FilterChips<T extends string>({
  options,
  selected,
  onSelect,
  label,
}: {
  options: ChipOption<T>[];
  selected: T;
  onSelect: (value: T) => void;
  /** Accessible name for the group, e.g. "Filter by status". */
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="filter-chips">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === selected}
          onClick={() => onSelect(o.value)}
        >
          {o.label}
          {o.count !== undefined && <span className="chip-count">{o.count}</span>}
        </button>
      ))}
    </div>
  );
}

/** Joined buttons: variant picker, enum fields, log/linear toggle. */
export function SegmentedControl<T extends string>({
  options,
  selected,
  onSelect,
  label,
  size = "normal",
}: {
  options: Array<{ value: T; label: string }>;
  selected: T;
  onSelect: (value: T) => void;
  label: string;
  /** `small` is the 32px-tall version used in chart headers. */
  size?: "normal" | "small";
}) {
  return (
    <div role="group" aria-label={label} className={`segmented segmented-${size}`}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === selected}
          onClick={() => onSelect(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
