import { useEffect, useState } from "react";
import { DiffChips } from "./badges";
import { SegmentedControl } from "./controls";

export interface FieldSchema {
  name: string;
  type: "int" | "float" | "bool" | "enum" | "str";
  is_optional: boolean;
  default: unknown;
  domain: string;
  step: number;
  choices: unknown[] | null;
}

export interface VariantSchema {
  name: string;
  description: string;
  default_config: Record<string, unknown>;
  fields: FieldSchema[];
}

export interface ExperimentSchema {
  description: string;
  variants: VariantSchema[];
}

export interface ExistingRunConfig {
  name: string;
  variant?: string;
  config: Record<string, unknown>;
}

interface Props {
  experiment: string;
  schema: ExperimentSchema;
  existingRuns: ExistingRunConfig[];
}

const NONE_SENTINEL = "__NONE__";

const SYMBOL_SET_LENGTHS: Record<string, number> = {
  all: 240,
  decimal: 10,
  hex: 16,
};

// ---------------------------------------------------------------------------
// Validation: a checklist instead of the first error
// ---------------------------------------------------------------------------

interface Check {
  ok: boolean;
  text: string;
}

interface ValidationResult {
  valid: boolean;
  /** Shown in the validity card: every failing check, plus the cross-field rules even when they pass. */
  checks: Check[];
  /** Derived values, e.g. "d_model=512 · pool=240". */
  summary?: string;
}

/** Per-field problems: half-typed numbers, None where not allowed, non-integers. */
function fieldChecks(
  fields: FieldSchema[],
  values: Record<string, unknown>,
  drafts: Record<string, string>,
): Check[] {
  const failures: Check[] = [];
  for (const field of fields) {
    if (field.name in drafts) {
      failures.push({ ok: false, text: `${field.name}: enter a valid ${field.type}` });
      continue;
    }
    const v = values[field.name];
    if (v === null && !field.is_optional) {
      failures.push({ ok: false, text: `${field.name} cannot be None` });
    } else if (field.type === "int" && v !== null && (typeof v !== "number" || !Number.isInteger(v))) {
      failures.push({ ok: false, text: `${field.name} must be an integer, got ${String(v)}` });
    }
  }
  return failures;
}

/**
 * The sir_yes_sir rules, mirroring `experiments/sir_yes_sir/config.py` (keep in sync).
 * Cross-field rules are always listed; single-field range rules only when they fail.
 */
function sirYesSirChecks(values: Record<string, unknown>): { checks: Check[]; summary: string } {
  const always: Check[] = [];
  const onlyIfFailing: Check[] = [];

  const symbolSet = String(values.symbol_set ?? "all");
  const fullPoolLen = SYMBOL_SET_LENGTHS[symbolSet] ?? 240;
  const symbolPoolSize = values.symbol_pool_size as number | null;
  always.push({
    ok: symbolPoolSize === null || (symbolPoolSize >= 2 && symbolPoolSize <= fullPoolLen),
    text:
      symbolPoolSize === null
        ? `symbol_pool_size None uses all of '${symbolSet}' (${fullPoolLen})`
        : `symbol_pool_size ${symbolPoolSize} within 2..len('${symbolSet}') ${fullPoolLen}`,
  });
  const poolLen = symbolPoolSize !== null ? symbolPoolSize : fullPoolLen;

  const alphabetSize = Number(values.alphabet_size);
  always.push({
    ok: alphabetSize >= 2 && alphabetSize <= poolLen,
    text: `alphabet_size ${alphabetSize} within 2..${poolLen} (symbol pool)`,
  });

  const embeddingMode = String(values.embedding_mode);
  const dPosition = Number(values.d_position);
  const usesPe = embeddingMode === "token_type_pe";
  always.push({
    ok: dPosition >= 0 && dPosition % 2 === 0 && (!usesPe || dPosition > 0),
    text: usesPe
      ? `d_position ${dPosition} is even and > 0 ('${embeddingMode}')`
      : `d_position ${dPosition} is an even int >= 0`,
  });

  const dScratch = Number(values.d_scratch);
  const fixedDModel = Boolean(values.fixed_d_model ?? true);
  const nHeads = Number(values.n_heads);
  const useTokenType = embeddingMode !== "nope";
  const dTokenType = useTokenType ? 16 : 0;
  const effectiveDPosition = usesPe ? dPosition : 0;
  const effectiveDScratch = fixedDModel
    ? dScratch + (useTokenType ? 0 : 16) + (usesPe ? 0 : dPosition)
    : dScratch;
  const dModel = 256 + dTokenType + effectiveDPosition + effectiveDScratch;
  always.push({
    ok: nHeads > 0 && dModel % nHeads === 0,
    text: `n_heads ${nHeads} divides d_model ${dModel}`,
  });

  const seed = Number(values.seed);
  const evalSeed = Number(values.eval_seed);
  always.push({ ok: seed !== evalSeed, text: `seed ${seed} ≠ eval_seed ${evalSeed}` });

  const atLeast = (name: string, min: number) => {
    const v = Number(values[name]);
    if (!(v >= min)) onlyIfFailing.push({ ok: false, text: `${name} must be >= ${min}, got ${v}` });
  };
  atLeast("x_length", 1);
  atLeast("d_scratch", 0);
  atLeast("n_layers", 1);
  atLeast("n_heads", 1);
  atLeast("d_ff", 1);
  atLeast("steps", 1);
  atLeast("batch_size", 1);
  atLeast("num_eval_problems", 1);
  atLeast("eval_batch_size", 1);
  const lr = Number(values.lr);
  if (!(lr > 0)) onlyIfFailing.push({ ok: false, text: `lr must be > 0, got ${lr}` });

  return { checks: [...onlyIfFailing, ...always], summary: `d_model=${dModel} · pool=${poolLen}` };
}

function validateConfig(
  expName: string,
  fields: FieldSchema[],
  values: Record<string, unknown>,
  drafts: Record<string, string>,
): ValidationResult {
  const failures = fieldChecks(fields, values, drafts);
  // Cross-field rules need every field to hold a real value first.
  if (failures.length > 0) return { valid: false, checks: failures };
  if (expName === "sir_yes_sir") {
    const { checks, summary } = sirYesSirChecks(values);
    return { valid: checks.every((c) => c.ok), checks, summary };
  }
  return { valid: true, checks: [] };
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

function formatCliFlag(field: FieldSchema, value: unknown): string {
  const flag = `--${field.name.replaceAll("_", "-")}`;
  if (value === null) {
    return `${flag} none`;
  }
  if (field.type === "bool" && !field.is_optional) {
    return value ? flag : `--no-${field.name.replaceAll("_", "-")}`;
  }
  return `${flag} ${String(value)}`;
}

function configsEqual(
  fields: FieldSchema[],
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): boolean {
  return fields.every((f) => a[f.name] === b[f.name]);
}

/** Variant defaults overlaid with a past run's values for the fields this variant has. */
function valuesFromRun(variant: VariantSchema, run: ExistingRunConfig): Record<string, unknown> {
  const next: Record<string, unknown> = { ...variant.default_config };
  for (const f of variant.fields) {
    if (f.name in run.config) next[f.name] = run.config[f.name];
  }
  return next;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ConfigBuilder({ experiment, schema, existingRuns }: Props) {
  const [variantName, setVariantName] = useState<string>(schema.variants[0]?.name ?? "");
  const [values, setValues] = useState<Record<string, unknown>>(() => ({
    ...(schema.variants[0]?.default_config ?? {}),
  }));
  // Holds raw text for any numeric input that is currently empty/mid-edit so
  // backspacing all digits works naturally without snapping back to the old number.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [startFrom, setStartFrom] = useState("");
  const [includeDefaults, setIncludeDefaults] = useState(false);
  const [copied, setCopied] = useState(false);

  // `?from=<run_name>` (from a run's "New run from this config") preloads that run.
  // Done after the first render so the server HTML and the browser's first render match.
  useEffect(() => {
    const from = new URLSearchParams(window.location.search).get("from");
    if (from) loadRun(from);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (schema.variants.length === 0) {
    return <p>No variants registered for this experiment.</p>;
  }

  const activeVariant = schema.variants.find((v) => v.name === variantName) ?? schema.variants[0];

  function selectVariant(name: string) {
    const target = schema.variants.find((v) => v.name === name) ?? schema.variants[0];
    setVariantName(target.name);
    // Keep the values of fields the new variant also has.
    setValues((current) => {
      const next: Record<string, unknown> = { ...target.default_config };
      for (const f of target.fields) {
        if (f.name in current) next[f.name] = current[f.name];
      }
      return next;
    });
    setStartFrom("");
    setCopied(false);
  }

  function loadRun(runName: string) {
    const run = existingRuns.find((r) => r.name === runName);
    if (!run) return;
    const variant = schema.variants.find((v) => v.name === run.variant) ?? schema.variants[0];
    setVariantName(variant.name);
    setValues(valuesFromRun(variant, run));
    setDrafts({});
    setStartFrom(runName);
    setCopied(false);
  }

  function setFieldValue(fieldName: string, value: unknown) {
    setValues((prev) => ({ ...prev, [fieldName]: value }));
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[fieldName];
      return next;
    });
    setCopied(false);
  }

  function handleNumericInput(field: FieldSchema, raw: string) {
    setCopied(false);
    // An optional number left empty means None.
    if (raw.trim() === "" && field.is_optional) {
      setFieldValue(field.name, null);
      return;
    }
    const parsed = Number(raw);
    if (raw.trim() === "" || Number.isNaN(parsed) || (field.type === "int" && !Number.isInteger(parsed))) {
      setDrafts((prev) => ({ ...prev, [field.name]: raw }));
      return;
    }
    setFieldValue(field.name, parsed);
  }

  function resetToDefaults() {
    setValues({ ...activeVariant.default_config });
    setDrafts({});
    setStartFrom("");
    setCopied(false);
  }

  const validation = validateConfig(experiment, activeVariant.fields, values, drafts);

  const isChanged = (f: FieldSchema) => f.name in drafts || values[f.name] !== f.default;
  const changed = activeVariant.fields
    .filter(isChanged)
    .map((f) => ({ name: f.name, value: f.name in drafts ? drafts[f.name] : values[f.name], default: f.default }));

  const flags = activeVariant.fields
    .filter((f) => includeDefaults || values[f.name] !== f.default)
    .map((f) => formatCliFlag(f, values[f.name]));
  const commandHead = `./yessir.sh run ${experiment} ${activeVariant.name}`;
  // Shown one flag per line; copied as a single line.
  const shownCommand = [commandHead, ...flags].join(" \\\n  ");
  const cliCommand = [commandHead, ...flags].join(" ");

  const matchingRuns = validation.valid
    ? existingRuns
        .filter((run) => run.variant === activeVariant.name && configsEqual(activeVariant.fields, values, run.config))
        .map((run) => run.name)
    : [];

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(cliCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access refused (e.g. not https): leave the label as it was.
    }
  }

  return (
    <div className="builder">
      <div className="panel builder-toolbar">
        <div className="toolbar-item">
          Variant
          <SegmentedControl
            label="Variant"
            options={schema.variants.map((v) => ({ value: v.name, label: v.name }))}
            selected={activeVariant.name}
            onSelect={selectVariant}
          />
        </div>
        <span className="muted variant-description">{activeVariant.description}</span>
        <label className="inline-select">
          Start from
          <select value={startFrom} onChange={(e) => (e.target.value ? loadRun(e.target.value) : resetToDefaults())}>
            <option value="">variant defaults</option>
            {existingRuns.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="button-secondary small-button" onClick={resetToDefaults}>
          Reset to defaults
        </button>
      </div>

      <div className="builder-columns">
        <div className="panel builder-form">
          <div className="builder-rows">
            {activeVariant.fields.map((field) => {
              const val = values[field.name];
              const hasDraft = field.name in drafts;
              const fieldChanged = isChanged(field);
              const id = `field-${field.name}`;
              return (
                <div key={field.name} className={`field-row${fieldChanged ? " is-changed" : ""}`}>
                  <label htmlFor={id} className="field-name">
                    <span className="field-dot" aria-hidden="true" />
                    {field.name}
                  </label>
                  <div className="field-control">
                    {field.choices !== null ? (
                      <SegmentedControl
                        label={field.name}
                        size="small"
                        options={[
                          ...field.choices.map((c) => ({ value: String(c), label: String(c) })),
                          ...(field.is_optional ? [{ value: NONE_SENTINEL, label: "None" }] : []),
                        ]}
                        selected={val === null ? NONE_SENTINEL : String(val)}
                        onSelect={(raw) => {
                          if (raw === NONE_SENTINEL) setFieldValue(field.name, null);
                          else setFieldValue(field.name, field.choices!.find((c) => String(c) === raw));
                        }}
                      />
                    ) : field.type === "int" || field.type === "float" ? (
                      <input
                        id={id}
                        type="number"
                        step={field.step}
                        placeholder={field.is_optional ? "None" : undefined}
                        value={hasDraft ? drafts[field.name] : typeof val === "number" ? val : ""}
                        onChange={(e) => handleNumericInput(field, e.target.value)}
                      />
                    ) : (
                      <input
                        id={id}
                        type="text"
                        value={val === null ? "" : String(val)}
                        onChange={(e) => setFieldValue(field.name, e.target.value)}
                      />
                    )}
                    {fieldChanged && (
                      <button
                        type="button"
                        className="field-reset"
                        aria-label={`Reset ${field.name} to default`}
                        onClick={() => setFieldValue(field.name, field.default)}
                      >
                        ↺ reset
                      </button>
                    )}
                  </div>
                  <span className="field-meta">default {field.default === null ? "None" : String(field.default)}</span>
                  <span className="field-meta">{field.domain}</span>
                </div>
              );
            })}
          </div>
        </div>

        <aside aria-label="Run summary" className="builder-side">
          <section className="panel side-card">
            <h2 className="section-heading">
              {changed.length} {changed.length === 1 ? "change" : "changes"} from {activeVariant.name} defaults
            </h2>
            {changed.length > 0 ? (
              <DiffChips changed={changed} />
            ) : (
              <span className="muted small-text">Nothing changed — this is the variant baseline.</span>
            )}
          </section>

          <section className={`side-card validity ${validation.valid ? "is-valid" : "is-invalid"}`}>
            <div className="validity-head">
              <h2 className="section-heading">{validation.valid ? "Valid" : "Invalid — fix before running"}</h2>
              {validation.summary && <span className="validity-summary">{validation.summary}</span>}
            </div>
            {validation.checks.length > 0 && (
              <ul className="checklist">
                {validation.checks.map((c, i) => (
                  <li key={i}>
                    <span aria-hidden="true" className={c.ok ? "check-pass" : "check-fail"}>
                      {c.ok ? "✓" : "✕"}
                    </span>
                    <span>
                      <span className="visually-hidden">{c.ok ? "Passes: " : "Fails: "}</span>
                      {c.text}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {validation.valid &&
            (matchingRuns.length > 0 ? (
              <section role="status" className="side-card duplicate-card">
                <h2 className="section-heading">Already run</h2>
                <p>
                  This exact config matches{" "}
                  {matchingRuns.map((name, i) => (
                    <span key={name}>
                      {i > 0 && ", "}
                      <a href={`/${experiment}/runs/${name}/`}>{name}</a>
                    </span>
                  ))}
                  . Change something, or rerun on purpose with a new seed.
                </p>
              </section>
            ) : (
              <section role="status" className="panel side-card new-config-card">
                <strong>New config</strong> — no existing run matches.
              </section>
            ))}

          <section className="side-card command-card">
            <div className="command-head">
              <h2 className="section-heading">Command</h2>
              <button type="button" onClick={copyCommand} disabled={!validation.valid}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre>{shownCommand}</pre>
            <label className="command-defaults">
              <input
                type="checkbox"
                checked={includeDefaults}
                onChange={(e) => {
                  setIncludeDefaults(e.target.checked);
                  setCopied(false);
                }}
              />
              Include default flags
            </label>
          </section>
        </aside>
      </div>
    </div>
  );
}
