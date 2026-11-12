"""Tests for `yessir.metrics` (`MetricsLogger`, `json_safe`)."""

import json
import math
from pathlib import Path

from yessir.metrics import MetricsLogger, json_safe


def test_json_safe_replaces_nan_and_inf():
    raw = {
        "loss": 0.25,
        "nan_rate": math.nan,
        "pos_inf": math.inf,
        "neg_inf": -math.inf,
        "count": 10,
    }
    cleaned = json_safe(raw)
    assert cleaned == {
        "loss": 0.25,
        "nan_rate": None,
        "pos_inf": None,
        "neg_inf": None,
        "count": 10,
    }
    # Must be accepted by strict JSON (allow_nan=False).
    assert json.loads(json.dumps(cleaned, allow_nan=False)) == cleaned


def test_metrics_logger_writes_valid_jsonl(tmp_path: Path):
    metrics_path = tmp_path / "run" / "metrics.jsonl"
    logger = MetricsLogger(metrics_path)
    logger.log(1, {"loss": 1.5, "undef": math.nan})
    logger.log(2, {"loss": 0.5, "exact_match": 1.0})

    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["step"] == 1
    assert rec1["loss"] == 1.5
    assert rec1["undef"] is None
    assert isinstance(rec1["elapsed_sec"], float)
    assert rec2["step"] == 2
    assert rec2["exact_match"] == 1.0
