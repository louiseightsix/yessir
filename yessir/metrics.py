"""Step-level JSONL metric logging (`MetricsLogger`, `json_safe`)."""

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def json_safe(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Replace non-finite floats (`NaN`, `±inf`) with `None` so JSON output is strictly valid."""
    cleaned: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, float) and not math.isfinite(val):
            cleaned[key] = None
        else:
            cleaned[key] = val
    return cleaned


class MetricsLogger:
    """Appends `NaN`-safe JSON Lines records (`metrics.jsonl`) during a training run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.monotonic()

    def log(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Append one JSON record with `step`, `elapsed_sec`, and `json_safe(metrics)`."""
        record: dict[str, Any] = {
            "step": step,
            "elapsed_sec": round(time.monotonic() - self._start_time, 3),
            **json_safe(metrics),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        return record
