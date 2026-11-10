"""Local JSONL metrics logger and structured console output.
Maintains 100% self-contained logging with zero external service dependencies.
"""

import json
import time
from pathlib import Path


def json_safe(metrics: dict) -> dict:
    """Replaces NaN (undefined rates, e.g. no carry columns in a cell) with None, so that records
    are standard JSON (null) rather than the bare NaN token."""
    return {k: None if isinstance(v, float) and v != v else v for k, v in metrics.items()}


class MetricsLogger:
    """Local JSONL metrics recorder for experiment tracking."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log(self, step: int, metrics: dict):
        """Append a JSON record to the log file."""
        record = {
            "step": step,
            "elapsed_sec": round(time.time() - self.start_time, 2),
            **json_safe(metrics),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def print_eval_summary(step: int, results: dict, train_bases: list[int]):
        """Print structured evaluation table to terminal (random | canonical axiom sets)."""
        bases = "{" + ", ".join(str(b) for b in train_bases) + "}"
        rows = [
            ("id", f"In-Domain       (B in {bases}, L 1..20)"),
            ("ood_base3", "Interpolation   (Base-3,  L 1..20)"),
            ("ood_base12", "Extrapolation   (Base-12, L 1..20)"),
            ("extrap_len", f"Length Extrap.  (B in {bases}, L 21..40)"),
            ("extrap_len_base3", "Length Extrap.  (Base-3,  L 21..40)"),
            ("extrap_len_base12", "Length Extrap.  (Base-12, L 21..40)"),
        ]

        def pct(key: str) -> str:
            value = results.get(key)
            return "  n/a " if value is None or value != value else f"{value * 100:5.1f}%"

        print("\n" + "=" * 86)
        print(f"EVALUATION SUMMARY | Step {step}   (random axioms | canonical axioms)")
        print("=" * 86)
        for cell, label in rows:
            step_val = results.get(f"{cell}_step_acc")
            if step_val is None:
                step_val = results.get(f"canon_{cell}_step_acc")
            step_acc = "  n/a " if step_val is None or step_val != step_val else f"{step_val * 100:5.1f}%"

            halt_val = results.get(f"{cell}_halt_acc")
            if halt_val is None:
                halt_val = results.get(f"canon_{cell}_halt_acc")
            halt_acc = "  n/a " if halt_val is None or halt_val != halt_val else f"{halt_val * 100:5.1f}%"

            print(
                f"  {label:<44} EM {pct(f'{cell}_em')} | {pct(f'canon_{cell}_em')}"
                f"   step {step_acc}   halt {halt_acc}"
            )
        print("=" * 86 + "\n")
