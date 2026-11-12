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
  const [includeDefaults, setIncludeDefaults] = useState(false);
  const [copied, setCopied] = useState(false);

  const schema = schemas[selectedExp];
  if (!schema) {
    return <p>No experiments registered.</p>;
  }

  const values = valuesByExp[selectedExp] ?? schema.default_config;

  function setFieldValue(fieldName: string, value: unknown) {
    setValuesByExp((prev) => ({
      ...prev,
      [selectedExp]: {
        ...(prev[selectedExp] ?? schema.default_config),
        [fieldName]: value,
      },
    }));
    setCopied(false);
  }

  function resetToDefaults() {
    setValuesByExp((prev) => ({
      ...prev,
      [selectedExp]: { ...schema.default_config },
    }));
    setCopied(false);
  }

  const flags = schema.fields
    .filter((f) => includeDefaults || values[f.name] !== f.default)
    .map((f) => formatCliFlag(f, values[f.name]));
  const cliCommand = `./yessir.sh run ${selectedExp}${flags.length > 0 ? " " + flags.join(" ") : ""}`;

  const matchingRuns = (existingRuns[selectedExp] ?? [])
    .filter((run) => configsEqual(schema.fields, values, run.config))
    .map((run) => run.name);

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
            const isNonDefault = val !== field.default;
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
                            checked={val === null}
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
                      {val !== null && (
                        <input
                          type="number"
                          step={field.step}
                          value={typeof val === "number" ? val : ""}
                          onChange={(e) => {
                            const raw = e.target.value;
                            if (raw === "") return;
                            const parsed =
                              field.type === "int"
                                ? Number.parseInt(raw, 10)
                                : Number.parseFloat(raw);
                            if (!Number.isNaN(parsed)) {
                              setFieldValue(field.name, parsed);
                            }
                          }}
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
          <button type="button" onClick={copyCommand}>
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
