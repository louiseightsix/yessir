#!/usr/bin/env python3
"""Aggregate and summarize all addition experiment results.

Scans .prototypes/transformadder/results/*/summary.json (and metrics.jsonl) and outputs a comprehensive
leaderboard and scientific breakdown across Paradigms, Tiers, and Regimes.

Usage (from the repo root):
    python .prototypes/transformadder/scripts/summarize_results.py
    python .prototypes/transformadder/scripts/summarize_results.py --format=markdown
"""

import argparse
import json
from pathlib import Path
import sys

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Addition experiment results aggregator")
    parser.add_argument("--experiments-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="Directory containing run directories")
    parser.add_argument("--format", choices=["table", "markdown", "json"], default="table", help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    return parser.parse_args()


def load_experiment_data(exp_dir: Path):
    summary_file = exp_dir / "summary.json"
    metrics_file = exp_dir / "metrics.jsonl"

    data = None
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    # Fallback to last line of metrics.jsonl if summary.json is not present
    if not data and metrics_file.exists():
        try:
            last_record = None
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_record = json.loads(line)
            if last_record:
                # Infer metadata from directory name
                parts = exp_dir.name.split("-")
                data = {
                    "variant": exp_dir.name,
                    "final_step": last_record.get("step", 0),
                    "results": last_record,
                    "inferred": True,
                }
        except Exception:
            pass

    return data


def format_pct(val):
    if val is None or val != val:
        return "-"
    return f"{val * 100:.1f}%"


def main():
    args = parse_args()
    exp_root = Path(args.experiments_dir)

    if not exp_root.exists():
        print(f"Error: {exp_root} directory not found.", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()])
    rows = []

    for d in exp_dirs:
        data = load_experiment_data(d)
        if not data:
            continue

        name = data.get("variant", d.name)
        step = data.get("final_step", 0)
        res = data.get("results", {})

        id_rand_em = res.get("id_em")
        id_canon_em = res.get("canon_id_em")
        id_step_acc = res.get("id_step_acc", res.get("canon_id_step_acc"))
        id_halt_acc = res.get("id_halt_acc", res.get("canon_id_halt_acc"))

        b3_rand_em = res.get("ood_base3_em")
        b3_canon_em = res.get("canon_ood_base3_em")

        b12_rand_em = res.get("ood_base12_em")
        b12_canon_em = res.get("canon_ood_base12_em")

        ext_id_em = res.get("extrap_len_em")
        ext_id_step = res.get("extrap_len_step_acc", res.get("canon_extrap_len_step_acc"))
        loss = res.get("loss")

        rows.append({
            "name": name,
            "step": step,
            "loss": f"{loss:.4f}" if isinstance(loss, (int, float)) else "-",
            "id_rand_em": format_pct(id_rand_em),
            "id_canon_em": format_pct(id_canon_em),
            "id_step_acc": format_pct(id_step_acc),
            "id_halt_acc": format_pct(id_halt_acc),
            "b3_rand_em": format_pct(b3_rand_em),
            "b3_canon_em": format_pct(b3_canon_em),
            "b12_rand_em": format_pct(b12_rand_em),
            "b12_canon_em": format_pct(b12_canon_em),
            "ext_id_step": format_pct(ext_id_step),
        })

    if not rows:
        print(f"No completed run summaries or metrics found in {exp_root}.", file=sys.stderr)
        print("Run the sync command first to pull results from Google Cloud Storage:", file=sys.stderr)
        print("  gcloud storage rsync -r $GCP_BUCKET/experiments/ .prototypes/transformadder/results/", file=sys.stderr)
        sys.exit(0)

    if args.format == "markdown":
        md = ["| Experiment Variant | Step | Loss | In-Domain EM (R/C) | Step Acc | Halt Acc | Base-3 EM (R/C) | Base-12 EM (R/C) | Extrap Len Step |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for r in rows:
            md.append(
                f"| `{r['name']}` | {r['step']} | {r['loss']} | {r['id_rand_em']} / {r['id_canon_em']} | {r['id_step_acc']} | {r['id_halt_acc']} | {r['b3_rand_em']} / {r['b3_canon_em']} | {r['b12_rand_em']} / {r['b12_canon_em']} | {r['ext_id_step']} |"
            )
        out_text = "\n".join(md)
    elif args.format == "json":
        out_text = json.dumps(rows, indent=2)
    else:
        # Table format
        header = f"{'Variant':<48} {'Step':>6} {'Loss':>7} {'ID EM (R/C)':>15} {'StepAcc':>8} {'HaltAcc':>8} {'Base3 EM':>13} {'Base12 EM':>13} {'ExtrapLen':>10}"
        lines = [header, "-" * len(header)]
        for r in rows:
            em_id = f"{r['id_rand_em']}/{r['id_canon_em']}"
            em_b3 = f"{r['b3_rand_em']}/{r['b3_canon_em']}"
            em_b12 = f"{r['b12_rand_em']}/{r['b12_canon_em']}"
            lines.append(
                f"{r['name']:<48} {r['step']:>6} {r['loss']:>7} {em_id:>15} {r['id_step_acc']:>8} {r['id_halt_acc']:>8} {em_b3:>13} {em_b12:>13} {r['ext_id_step']:>10}"
            )
        out_text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(out_text + "\n", encoding="utf-8")
        print(f"Summary written to {args.output}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
