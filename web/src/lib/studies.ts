// Derived values for studies, shared by the Studies list and Study detail pages.
import type { ExperimentData, RunData, StudyData } from "./data";

export interface StudyEntry {
  runName: string;
  label: string | null;
  /** null when the study cites a run that isn't in results/. */
  run: RunData | null;
}

/** The study's runs in the order the study file lists them. */
export function resolveStudyRuns(exp: ExperimentData, study: StudyData): StudyEntry[] {
  return study.runs.map((r) => ({
    runName: r.run_name,
    label: r.label,
    run: exp.runsByName.get(r.run_name) ?? null,
  }));
}

/** "Valid" for bests and charts: evaluated and not flawed. */
export function isValid(run: RunData | null): run is RunData {
  return run !== null && run.status === "evaluated";
}

// Seeds often differ between runs of a sweep, so they never count as the swept setting.
const IGNORED_FOR_SWEEP = new Set(["seed", "eval_seed"]);

/**
 * The one setting (other than the seeds) whose value differs across the study's
 * runs, or null if none or several do. Used for "sweeps <param>" and the sweep
 * chart's x axis.
 */
export function sweptSetting(runs: RunData[]): string | null {
  if (runs.length < 2) return null;
  const keys = new Set(runs.flatMap((r) => Object.keys(r.config)));
  const varying = [...keys].filter((k) => {
    if (IGNORED_FOR_SWEEP.has(k)) return false;
    const values = new Set(runs.map((r) => JSON.stringify(r.config[k])));
    return values.size > 1;
  });
  return varying.length === 1 ? varying[0] : null;
}

/** Newest run date (from the run name) or, with no runs, the file's modification date. */
export function studyUpdated(entries: StudyEntry[], study: StudyData): string {
  const dates = entries.map((e) => e.runName.slice(0, 10)).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d));
  return dates.length > 0 ? dates.sort().at(-1)! : study.fileModified;
}
