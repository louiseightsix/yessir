import { useState } from "react";

export interface FieldSchema {
  name: string;
  type: "int" | "float" | "bool" | "enum" | "str";
  is_optional: boolean;
  default: unknown;
  domain: string;
  step: number;
  choices: unknown[] | null;
}

export interface ExperimentSchema {
  description: string;
  default_config: Record<string, unknown>;
  fields: FieldSchema[];
}

export interface ExistingRunConfig {
  name: string;
  config: Record<string, unknown>;
}

interface Props {
  schemas: Record<string, ExperimentSchema>;
  existingRuns: Record<string, ExistingRunConfig[]>;
}

const NONE_SENTINEL = "__NONE__";

const SYMBOL_SET_LENGTHS: Record<string, number> = {
  all: 250,
  decimal: 10,
  hex: 16,
};

interface ValidationResult {
  valid: boolean;
  summary?: string;
  error?: string;
}

function validateConfig(
  expName: string,
  fields: FieldSchema[],
  values: Record<string, unknown>,
  drafts: Record<string, string>,
): ValidationResult {
  for (const field of fields) {
    if (field.name in drafts) {
      return {
        valid: false,
        error: `${field.name}: enter a valid ${field.type}`,
      };
    }
    const v = values[field.name];
    if (v === null && !field.is_optional) {
      return { valid: false, error: `${field.name} cannot be None` };
    }
    if (field.type === "int" && v !== null && (!Number.isInteger(v) || typeof v !== "number")) {
      return { valid: false, error: `${field.name} must be an integer, got ${String(v)}` };
    }
  }

  if (expName === "sir_yes_sir") {
    const symbolSet = String(values.symbol_set ?? "all");
    const fullPoolLen = SYMBOL_SET_LENGTHS[symbolSet] ?? 250;
    const symbolPoolSize = values.symbol_pool_size as number | null;
    if (
      symbolPoolSize !== null &&
      (symbolPoolSize < 2 || symbolPoolSize > fullPoolLen)
    ) {
      return {
        valid: false,
        error: `symbol_pool_size must be in 2..${fullPoolLen} for symbol_set='${symbolSet}', got ${symbolPoolSize}`,
      };
    }
    const poolLen = symbolPoolSize !== null ? symbolPoolSize : fullPoolLen;
    const alphabetSize = Number(values.alphabet_size);
    if (alphabetSize < 2 || alphabetSize > poolLen) {
      return {
        valid: false,
        error: `alphabet_size must be in 2..${poolLen} (size of symbol_pool), got ${alphabetSize}`,
      };
    }
    const xLength = Number(values.x_length);
    if (xLength < 1) {
      return { valid: false, error: `x_length must be >= 1, got ${xLength}` };
    }
    const dPosition = Number(values.d_position);
    if (dPosition < 0 || dPosition % 2 !== 0) {
      return {
        valid: false,
        error: `d_position must be a non-negative even int, got ${dPosition}`,
      };
    }
    const embeddingMode = String(values.embedding_mode);
    if (embeddingMode === "token_type_pe" && dPosition === 0) {
      return {
        valid: false,
        error: "d_position must be > 0 when embedding_mode is 'token_type_pe'",
      };
    }
    if (embeddingMode !== "token_type_pe" && dPosition !== 0) {
      return {
        valid: false,
        error: `d_position must be 0 when embedding_mode is '${embeddingMode}', got ${dPosition}`,
      };
    }
    const dScratch = Number(values.d_scratch);
    if (dScratch < 0) {
      return { valid: false, error: `d_scratch must be >= 0, got ${dScratch}` };
    }
    const nLayers = Number(values.n_layers);
    if (nLayers <= 0) {
      return { valid: false, error: `n_layers must be > 0, got ${nLayers}` };
    }
    const nHeads = Number(values.n_heads);
    if (nHeads <= 0) {
      return { valid: false, error: `n_heads must be > 0, got ${nHeads}` };
    }
    const dTokenType = embeddingMode !== "nope" ? 16 : 0;
    const dModel = 256 + dTokenType + dPosition + dScratch;
    if (dModel % nHeads !== 0) {
      return {
        valid: false,
        error: `d_model (${dModel}) must be divisible by n_heads (${nHeads})`,
      };
    }
    const dFf = Number(values.d_ff);
    if (dFf <= 0) {
      return { valid: false, error: `d_ff must be > 0, got ${dFf}` };
    }
    const steps = Number(values.steps);
    if (steps < 1) {
      return { valid: false, error: `steps must be >= 1, got ${steps}` };
    }
    const batchSize = Number(values.batch_size);
    if (batchSize < 1) {
      return { valid: false, error: `batch_size must be >= 1, got ${batchSize}` };
    }
    const lr = Number(values.lr);
    if (!(lr > 0)) {
      return { valid: false, error: `lr must be > 0, got ${lr}` };
    }
    const numEvalProblems = Number(values.num_eval_problems);
    if (numEvalProblems < 1) {
      return {
        valid: false,
        error: `num_eval_problems must be >= 1, got ${numEvalProblems}`,
      };
    }
    const evalBatchSize = Number(values.eval_batch_size);
    if (evalBatchSize < 1) {
      return {
        valid: false,
        error: `eval_batch_size must be >= 1, got ${evalBatchSize}`,
      };
    }
    const seed = Number(values.seed);
    const evalSeed = Number(values.eval_seed);
    if (evalSeed === seed) {
      return {
        valid: false,
        error: `eval_seed (${evalSeed}) must differ from training seed (${seed})`,
      };
    }
    return {
      valid: true,
      summary: `d_model=${dModel}, symbol_pool_size=${poolLen}`,
    };
  }

  return { valid: true };
}

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

export default function ConfigBuilder({ schemas, existingRuns }: Props) {
  const expNames = Object.keys(schemas);
  const [selectedExp, setSelectedExp] = useState<string>(expNames[0] ?? "");
  const [valuesByExp, setValuesByExp] = useState<Record<string, Record<string, unknown>>>(() =>
    Object.fromEntries(
      Object.entries(schemas).map(([name, schema]) => [name, { ...schema.default_config }]),
    ),
  );
  // Holds raw text for any numeric input that is currently empty/mid-edit so
  // backspacing all digits works naturally without snapping back to the old number.
  const [draftsByExp, setDraftsByExp] = useState<Record<string, Record<string, string>>>({});
  const [includeDefaults, setIncludeDefaults] = useState(false);
  const [copied, setCopied] = useState(false);

  const schema = schemas[selectedExp];
  if (!schema) {
    return <p>No experiments registered.</p>;
  }

  const values = valuesByExp[selectedExp] ?? schema.default_config;
  const drafts = draftsByExp[selectedExp] ?? {};

  function setFieldValue(fieldName: string, value: unknown) {
    setValuesByExp((prev) => ({
      ...prev,
      [selectedExp]: {
        ...(prev[selectedExp] ?? schema.default_config),
        [fieldName]: value,
      },
    }));
    setDraftsByExp((prev) => {
      const expDrafts = { ...(prev[selectedExp] ?? {}) };
      delete expDrafts[fieldName];
      return { ...prev, [selectedExp]: expDrafts };
    });
    setCopied(false);
  }

  function handleNumericInput(field: FieldSchema, raw: string) {
    setCopied(false);
    if (raw.trim() === "") {
      setDraftsByExp((prev) => ({
        ...prev,
        [selectedExp]: { ...(prev[selectedExp] ?? {}), [field.name]: raw },
      }));
      return;
    }
    const parsed = Number(raw);
    if (Number.isNaN(parsed) || (field.type === "int" && !Number.isInteger(parsed))) {
      setDraftsByExp((prev) => ({
        ...prev,
        [selectedExp]: { ...(prev[selectedExp] ?? {}), [field.name]: raw },
      }));
      return;
    }
    setFieldValue(field.name, parsed);
  }

  function resetToDefaults() {
    setValuesByExp((prev) => ({
      ...prev,
      [selectedExp]: { ...schema.default_config },
    }));
    setDraftsByExp((prev) => ({ ...prev, [selectedExp]: {} }));
    setCopied(false);
  }

  const validation = validateConfig(selectedExp, schema.fields, values, drafts);

  const flags = schema.fields
    .filter((f) => includeDefaults || values[f.name] !== f.default)
    .map((f) => formatCliFlag(f, values[f.name]));
  const cliCommand = `./yessir.sh run ${selectedExp}${flags.length > 0 ? " " + flags.join(" ") : ""}`;

  const matchingRuns = validation.valid
    ? (existingRuns[selectedExp] ?? [])
        .filter((run) => configsEqual(schema.fields, values, run.config))
        .map((run) => run.name)
    : [];

  async function copyCommand() {
    await navigator.clipboard.writeText(cliCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="config-builder">
      <div className="builder-toolbar">
        <label>
          <strong>Experiment: </strong>
          <select
            value={selectedExp}
            onChange={(e) => {
              setSelectedExp(e.target.value);
              setCopied(false);
            }}
          >
            {expNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={resetToDefaults}>
          Reset to defaults
        </button>
      </div>

      <p className="meta">{schema.description}</p>

      <table>
        <thead>
          <tr>
            <th>Setting</th>
            <th>Value</th>
            <th>Default</th>
            <th>Allowed / Range</th>
          </tr>
        </thead>
        <tbody>
          {schema.fields.map((field) => {
            const val = values[field.name];
            const hasDraft = field.name in drafts;
            const isNonDefault = hasDraft || val !== field.default;
            return (
              <tr key={field.name} className={isNonDefault ? "non-default" : undefined}>
                <th>
                  <code>{field.name}</code>
                </th>
                <td>
                  {field.choices !== null ? (
                    <select
                      value={val === null ? NONE_SENTINEL : String(val)}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === NONE_SENTINEL) {
                          setFieldValue(field.name, null);
                        } else if (field.type === "bool") {
                          setFieldValue(field.name, raw === "true");
                        } else {
                          setFieldValue(field.name, raw);
                        }
                      }}
                    >
                      {field.choices.map((choice) => (
                        <option key={String(choice)} value={String(choice)}>
                          {String(choice)}
                        </option>
                      ))}
                      {field.is_optional && <option value={NONE_SENTINEL}>None</option>}
                    </select>
                  ) : field.type === "int" || field.type === "float" ? (
                    <span className="numeric-control">
                      {field.is_optional && (
                        <label className="none-toggle">
                          <input
                            type="checkbox"
                            checked={!hasDraft && val === null}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setFieldValue(field.name, null);
                              } else {
                                setFieldValue(
                                  field.name,
                                  typeof field.default === "number" ? field.default : 2,
                                );
                              }
                            }}
                          />{" "}
                          None
                        </label>
                      )}
                      {(hasDraft || val !== null) && (
                        <input
                          type="number"
                          step={field.step}
                          value={hasDraft ? drafts[field.name] : typeof val === "number" ? val : ""}
                          onChange={(e) => handleNumericInput(field, e.target.value)}
                        />
                      )}
                    </span>
                  ) : (
                    <input
                      type="text"
                      value={val === null ? "" : String(val)}
                      onChange={(e) => setFieldValue(field.name, e.target.value)}
                    />
                  )}
                </td>
                <td>
                  <code>{field.default === null ? "None" : String(field.default)}</code>
                </td>
                <td>
                  <code>{field.domain}</code>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {validation.valid ? (
        <p className="status-valid">
          <strong>Status: VALID</strong>
          {validation.summary ? ` (${validation.summary})` : ""}
        </p>
      ) : (
        <p className="status-invalid">
          <strong>Status: INVALID</strong> — {validation.error}
        </p>
      )}

      {matchingRuns.length > 0 && (
        <p className="duplicate-warning">
          Matches existing {matchingRuns.length === 1 ? "run" : "runs"}:{" "}
          <code>{matchingRuns.join(", ")}</code>
        </p>
      )}

      <div className="cli-output">
        <div className="cli-header">
          <strong>CLI command</strong>
          <label className="meta">
            <input
              type="checkbox"
              checked={includeDefaults}
              onChange={(e) => setIncludeDefaults(e.target.checked)}
            />{" "}
            Include default flags
          </label>
          <button type="button" onClick={copyCommand} disabled={!validation.valid}>
            {copied ? "Copied!" : "Copy command"}
          </button>
        </div>
        <pre>
          <code>{cliCommand}</code>
        </pre>
      </div>
    </div>
  );
}
