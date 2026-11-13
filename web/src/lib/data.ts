// Reads `results/`, `studies/` and the experiment schemas from disk.
// Runs at build time only (and on each request under `astro dev`).
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import type { ExperimentSchema, VariantSchema } from "../components/ConfigBuilder";

// Astro runs with web/ as the working directory; the repo root is one up.
export const repoRoot = resolve(process.cwd(), "..");
const resultsDir = resolve(repoRoot, "results");
const studiesDir = resolve(repoRoot, "studies");

// Plain python3, not `uv run`: the export needs only the standard library,
// so the build doesn't have to install torch. YESSIR_PYTHON overrides it.
const python = process.env.YESSIR_PYTHON ?? "python3";

// ---------------------------------------------------------------------------
// Raw file shapes
// ---------------------------------------------------------------------------

export interface ConfigFile {
  experiment?: string;
  variant?: string;
  device?: string;
  git_commit_sha?: string;
  git_short_sha?: string;
  git_commit_date?: string;
  git_dirty?: boolean;
  config?: Record<string, unknown>;
}

export interface TrainPoint {
  step: number;
  elapsed_sec: number;
  loss: number;
}

// Keep in sync with `Study` / `StudyRun` in `yessir/artefacts.py`.
interface StudyRunEntry {
  run_name: string;
  label?: string | null;
}

interface StudyFile {
  title?: string;
  status?: string;
  kind?: string | null;
  question?: string | null;
  runs?: StudyRunEntry[];
  conclusion?: string | null;
}

// ---------------------------------------------------------------------------
// Loaded and derived shapes
// ---------------------------------------------------------------------------

/** Flawed wins over everything; otherwise a run without `eval.json` is still in progress. */
export type RunStatus = "evaluated" | "in_progress" | "flawed";

/** One setting whose value differs from the run's variant default. */
export interface ChangedSetting {
  name: string;
  value: unknown;
  default: unknown;
}

export interface RunData {
  name: string;
  meta: ConfigFile | null;
  evalMetrics: Record<string, number> | null;
  trainPoints: TrainPoint[];
  readmeBody: string | null;
  flawedReason: string | null;
  /** Reason from an `OUTDATED:` first line: sound under an earlier spec, not the current one. */
  outdatedReason: string | null;
  // Derived:
  status: RunStatus;
  variant: string | null;
  config: Record<string, unknown>;
  changed: ChangedSetting[];
  /** `config.steps`, if the run recorded it. */
  totalSteps: number | null;
  /** When `config.json` was written (ms since epoch), to order runs from the same day. */
  startedAtMs: number;
}

export interface StudyData {
  slug: string;
  title: string;
  status: string;
  kind: string | null;
  question: string | null;
  runs: Array<{ run_name: string; label: string | null }>;
  conclusion: string | null;
  /** The study file's modification date (YYYY-MM-DD), for studies with no runs yet. */
  fileModified: string;
}

export interface StudyRef {
  slug: string;
  title: string;
  label: string | null;
}

export interface ExperimentData {
  name: string;
  schema: ExperimentSchema | null;
  /** Newest first: see `compareNewestFirst`. */
  runs: RunData[];
  studies: StudyData[];
  runsByName: Map<string, RunData>;
  /** Reverse index: run name → the studies that cite it. */
  studiesByRun: Map<string, StudyRef[]>;
  /** The metric that drives run bars and sweep charts: the first key in `eval.json`. */
  headlineMetric: string | null;
  /** Union of eval metric names across runs, in first-seen order. */
  metricNames: string[];
}

// ---------------------------------------------------------------------------
// File readers (unchanged from the old single-page browser)
// ---------------------------------------------------------------------------

function readJsonIfExists<T>(path: string): T | null {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch {
    return null;
  }
}

export function readTrainJsonl(path: string): TrainPoint[] {
  if (!existsSync(path)) return [];
  const points: TrainPoint[] = [];
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      points.push(JSON.parse(trimmed) as TrainPoint);
    } catch {
      // Ignore a trailing half-written line while a run is in progress.
    }
  }
  return points;
}

function listSubdirs(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
}

/** Default config for a run's variant, falling back to the first variant. */
export function variantDefaults(
  schema: ExperimentSchema | null,
  variant: string | null | undefined,
): Record<string, unknown> {
  const variants: VariantSchema[] = schema?.variants ?? [];
  return (
    variants.find((v) => v.name === variant)?.default_config ??
    variants[0]?.default_config ??
    {}
  );
}

export function loadRun(
  runDir: string,
  name: string,
  schema: ExperimentSchema | null,
): RunData {
  const meta = readJsonIfExists<ConfigFile>(resolve(runDir, "config.json"));
  const evalMetrics = readJsonIfExists<Record<string, number>>(resolve(runDir, "eval.json"));
  const trainPoints = readTrainJsonl(resolve(runDir, "train.jsonl"));
  const readmePath = resolve(runDir, "README.md");
  const rawReadme = existsSync(readmePath) ? readFileSync(readmePath, "utf8").trim() : null;
  const lines = rawReadme ? rawReadme.split("\n") : [];
  const firstLine = lines[0] ?? "";
  const isFlawed = firstLine.startsWith("FLAWED:");
  const flawedReason = isFlawed ? firstLine.slice("FLAWED:".length).trim() : null;
  const isOutdated = firstLine.startsWith("OUTDATED:");
  const outdatedReason = isOutdated ? firstLine.slice("OUTDATED:".length).trim() : null;
  const remaining = (isFlawed || isOutdated ? lines.slice(1) : lines).join("\n").trim();
  const readmeBody = remaining.length > 0 ? remaining : null;

  const variant = meta?.variant ?? null;
  const config = meta?.config ?? {};
  const defaults = variantDefaults(schema, variant);
  const changed = Object.entries(config)
    .filter(([k, v]) => k in defaults && defaults[k] !== v)
    .map(([k, v]) => ({ name: k, value: v, default: defaults[k] }));
  const status: RunStatus =
    flawedReason !== null ? "flawed" : evalMetrics === null ? "in_progress" : "evaluated";

  return {
    name,
    meta,
    evalMetrics,
    trainPoints,
    readmeBody,
    flawedReason,
    outdatedReason,
    status,
    variant,
    config,
    changed,
    totalSteps: typeof config.steps === "number" ? config.steps : null,
    startedAtMs: fileMtimeMs(resolve(runDir, "config.json")),
  };
}

/** YYYY-MM-DD in local time, matching the dates in run names. */
function localDate(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function fileMtimeMs(path: string): number {
  return existsSync(path) ? statSync(path).mtimeMs : 0;
}

/**
 * Newest first. Run names start with the date (`2026-11-12-…`) but not the time,
 * so runs from the same day are ordered by when their `config.json` was written.
 * (A fresh git checkout resets mtimes, which only affects order within a day.)
 */
function compareNewestFirst(a: RunData, b: RunData): number {
  const byDate = b.name.slice(0, 10).localeCompare(a.name.slice(0, 10));
  if (byDate !== 0) return byDate;
  return b.startedAtMs - a.startedAtMs || b.name.localeCompare(a.name);
}

export function loadStudies(expStudiesDir: string): StudyData[] {
  if (!existsSync(expStudiesDir)) return [];
  return readdirSync(expStudiesDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .sort((a, b) => a.name.localeCompare(b.name))
    .flatMap((entry) => {
      const raw = readJsonIfExists<StudyFile>(resolve(expStudiesDir, entry.name));
      if (!raw || typeof raw.title !== "string") return [];
      const slug = entry.name.slice(0, -".json".length);
      const runs = Array.isArray(raw.runs)
        ? raw.runs
            .filter((r): r is StudyRunEntry => Boolean(r && typeof r.run_name === "string"))
            .map((r) => ({
              run_name: r.run_name,
              label: typeof r.label === "string" ? r.label : null,
            }))
        : [];
      return [
        {
          slug,
          title: raw.title,
          status: typeof raw.status === "string" ? raw.status : "draft",
          kind: typeof raw.kind === "string" ? raw.kind : null,
          question: typeof raw.question === "string" ? raw.question : null,
          runs,
          conclusion: typeof raw.conclusion === "string" ? raw.conclusion : null,
          fileModified: localDate(fileMtimeMs(resolve(expStudiesDir, entry.name))),
        },
      ];
    });
}

// ---------------------------------------------------------------------------
// Whole-repo loaders
// ---------------------------------------------------------------------------

function loadSchemas(): Record<string, ExperimentSchema> {
  return JSON.parse(
    execFileSync(python, ["-m", "experiments.web_export"], { cwd: repoRoot, encoding: "utf8" }),
  );
}

function loadExperiment(name: string, schema: ExperimentSchema | null): ExperimentData {
  const expResultsDir = resolve(resultsDir, name);
  const runs = listSubdirs(expResultsDir)
    .map((runName) => loadRun(resolve(expResultsDir, runName), runName, schema))
    .sort(compareNewestFirst);
  const studies = loadStudies(resolve(studiesDir, name));
  const runsByName = new Map(runs.map((r) => [r.name, r]));
  const studiesByRun = new Map<string, StudyRef[]>();
  for (const study of studies) {
    for (const entry of study.runs) {
      const list = studiesByRun.get(entry.run_name) ?? [];
      list.push({ slug: study.slug, title: study.title, label: entry.label });
      studiesByRun.set(entry.run_name, list);
    }
  }

  const metricNames: string[] = [];
  for (const run of [...runs].reverse()) {
    for (const key of Object.keys(run.evalMetrics ?? {})) {
      if (!metricNames.includes(key)) metricNames.push(key);
    }
  }

  return {
    name,
    schema,
    runs,
    studies,
    runsByName,
    studiesByRun,
    headlineMetric: metricNames[0] ?? null,
    metricNames,
  };
}

export interface SiteData {
  experiments: ExperimentData[];
  schemas: Record<string, ExperimentSchema>;
  pythonVersion: string;
  headShortSha: string | null;
}

function readSiteData(): SiteData {
  const schemas = loadSchemas();
  const names = Array.from(
    new Set([...Object.keys(schemas), ...listSubdirs(resultsDir), ...listSubdirs(studiesDir)]),
  ).sort((a, b) => a.localeCompare(b));
  const experiments = names.map((name) => loadExperiment(name, schemas[name] ?? null));

  // "Python 3.12.13" → "3.12.13"
  const pythonVersion = execFileSync(python, ["--version"], { encoding: "utf8" })
    .trim()
    .replace(/^Python\s+/, "");
  let headShortSha: string | null = null;
  try {
    headShortSha = execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
    }).trim();
  } catch {
    // Not a git checkout (e.g. a deployed copy): leave the footer without a SHA.
  }

  return { experiments, schemas, pythonVersion, headShortSha };
}

// A static build renders one page per run and study, so read the disk once.
// Under `astro dev` re-read on every request: the dev server reloads the page
// when `results/` changes, and a cache would serve the old data.
let cached: SiteData | null = null;

export function getSiteData(): SiteData {
  if (import.meta.env.DEV) return readSiteData();
  cached ??= readSiteData();
  return cached;
}

export function getExperiment(name: string): ExperimentData {
  const exp = getSiteData().experiments.find((e) => e.name === name);
  if (!exp) throw new Error(`unknown experiment: ${name}`);
  return exp;
}
