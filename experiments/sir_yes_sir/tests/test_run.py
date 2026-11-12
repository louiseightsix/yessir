"""Tests for `experiments.sir_yes_sir.run` and the top-level `experiments` CLI dispatcher."""

import json
from pathlib import Path

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.__main__ import main as experiments_main
from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.run import SIR_YES_SIR_EXPERIMENT
from yessir.experiment import run_experiments_cli


def test_sir_yes_sir_registered_in_top_level_experiments():
    names = [exp.name for exp in EXPERIMENTS]
    assert "sir_yes_sir" in names
    assert SIR_YES_SIR_EXPERIMENT in EXPERIMENTS


def test_dispatch_sir_yes_sir_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-12-953e4d9-amber-falcon"
    summary = run_experiments_cli(
        EXPERIMENTS,
        [
            "sir_yes_sir",
            "--out-dir",
            str(run_dir),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--eval-problems",
            "4",
            "--n-layers",
            "1",
            "--n-heads",
            "4",
            "--d-ff",
            "32",
            "--d-position",
            "8",
            "--d-scratch",
            "0",
        ],
    )
    assert summary is not None
    assert set(summary.keys()) == {
        "final_loss",
        "exact_match",
        "step_accuracy",
        "halt_accuracy",
    }
    assert isinstance(summary["final_loss"], float)
    assert 0.0 <= summary["exact_match"] <= 1.0
    assert 0.0 <= summary["step_accuracy"] <= 1.0
    assert 0.0 <= summary["halt_accuracy"] <= 1.0

    # Verify all three Stage A step 9 artefacts were written.
    config_file = run_dir / "config.json"
    metrics_file = run_dir / "metrics.jsonl"
    summary_file = run_dir / "summary.json"
    assert config_file.is_file()
    assert metrics_file.is_file()
    assert summary_file.is_file()

    loaded_cfg = SirYesSirConfig.from_dict(json.loads(config_file.read_text()))
    assert loaded_cfg.steps == 2
    assert loaded_cfg.n_layers == 1

    metric_lines = [json.loads(line) for line in metrics_file.read_text().splitlines()]
    assert len(metric_lines) == 2
    assert metric_lines[0]["step"] == 1
    assert metric_lines[1]["step"] == 2
    assert metric_lines[1]["exact_match"] == summary["exact_match"]

    loaded_summary = json.loads(summary_file.read_text())
    assert loaded_summary["experiment"] == "sir_yes_sir"
    assert loaded_summary["run_name"] == "2026-11-12-953e4d9-amber-falcon"
    assert loaded_summary["metrics"] == summary

    out = capsys.readouterr().out
    assert "[sir_yes_sir] step=2" in out
    assert f"artefacts written to {run_dir}" in out


def test_experiments_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]):
    experiments_main([])
    out = capsys.readouterr().out
    assert "sir_yes_sir" in out
