"""Tests for `experiments.sir_yes_sir.run` and the top-level `experiments` CLI dispatcher."""

import json
from pathlib import Path

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.__main__ import main as experiments_main
from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.run import SIR_YES_SIR_EXPERIMENT
from experiments.sir_yes_sir.variants.ar.config import SirYesSirArConfig
from yessir.experiment import run_experiments_cli


def test_sir_yes_sir_registered_in_top_level_experiments():
    names = [exp.name for exp in EXPERIMENTS]
    assert "sir_yes_sir" in names
    assert SIR_YES_SIR_EXPERIMENT in EXPERIMENTS
    assert [v.name for v in SIR_YES_SIR_EXPERIMENT.variants] == ["sm_mh", "ar"]


def test_dispatch_sir_yes_sir_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-12-953e4d9-amber-falcon"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "sir_yes_sir",
            "sm_mh",
            "--out-dir",
            str(run_dir),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--num-eval-problems",
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
    assert eval_results is not None
    assert set(eval_results.keys()) == {
        "exact_match",
        "step_accuracy",
        "halt_accuracy",
    }
    assert 0.0 <= eval_results["exact_match"] <= 1.0
    assert 0.0 <= eval_results["step_accuracy"] <= 1.0
    assert 0.0 <= eval_results["halt_accuracy"] <= 1.0

    # Verify config.json, train.jsonl, and eval.json were written.
    config_file = run_dir / "config.json"
    train_file = run_dir / "train.jsonl"
    eval_file = run_dir / "eval.json"
    assert config_file.is_file()
    assert train_file.is_file()
    assert eval_file.is_file()

    config_payload = json.loads(config_file.read_text())
    assert config_payload["experiment"] == "sir_yes_sir"
    assert config_payload["variant"] == "sm_mh"
    assert "git_commit_sha" in config_payload
    loaded_cfg = SirYesSirConfig.from_dict(config_payload["config"])
    assert loaded_cfg.steps == 2
    assert loaded_cfg.n_layers == 1

    train_lines = [json.loads(line) for line in train_file.read_text().splitlines()]
    assert len(train_lines) == 2
    assert train_lines[0]["step"] == 1
    assert isinstance(train_lines[0]["loss"], float)
    assert train_lines[1]["step"] == 2
    assert isinstance(train_lines[1]["loss"], float)

    loaded_eval = json.loads(eval_file.read_text())
    assert loaded_eval == eval_results

    out = capsys.readouterr().out
    assert "[sir_yes_sir:sm_mh] Starting run: B=4, L=3" in out
    assert "device=cpu" in out
    assert "[sir_yes_sir:sm_mh] step=1/2 loss=" in out
    assert "[sir_yes_sir:sm_mh] step=2/2 loss=" in out
    assert "[sir_yes_sir:sm_mh] eval (4 problems):" in out
    assert f"artefacts written to {run_dir}" in out


def test_experiments_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]):
    experiments_main([])
    out = capsys.readouterr().out
    assert "sir_yes_sir" in out


def test_dispatch_sir_yes_sir_ar_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-12-953e4d9-quiet-otter"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "sir_yes_sir",
            "ar",
            "--out-dir",
            str(run_dir),
            "--step-order",
            "wrap_first",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--num-eval-problems",
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
    assert eval_results is not None
    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "sir_yes_sir"
    assert config_payload["variant"] == "ar"
    loaded_cfg = SirYesSirArConfig.from_dict(config_payload["config"])
    assert loaded_cfg.step_order.value == "wrap_first"
    out = capsys.readouterr().out
    assert "[sir_yes_sir:ar] Starting run:" in out
