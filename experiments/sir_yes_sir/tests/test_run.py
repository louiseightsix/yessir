"""Tests for `experiments.sir_yes_sir.run` and the top-level `experiments` CLI dispatcher."""

import json
from pathlib import Path

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.__main__ import main as experiments_main
from experiments.sir_yes_sir.config import SirYesSirClsConfig, SirYesSirConfig
from experiments.sir_yes_sir.run import SIR_YES_SIR_EXPERIMENT
from yessir.experiment import run_experiments_cli


def test_sir_yes_sir_registered_in_top_level_experiments():
    names = [exp.name for exp in EXPERIMENTS]
    assert "sir_yes_sir" in names
    assert SIR_YES_SIR_EXPERIMENT in EXPERIMENTS
    assert [v.name for v in SIR_YES_SIR_EXPERIMENT.variants] == ["cls", "ar"]


def test_dispatch_sir_yes_sir_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-12-953e4d9-amber-falcon"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "sir_yes_sir",
            "cls",
            "--out-dir",
            str(run_dir),
            "--max-steps",
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

    # Verify config.json, train.jsonl, eval.jsonl, and eval.json were written.
    config_file = run_dir / "config.json"
    train_file = run_dir / "train.jsonl"
    eval_jsonl_file = run_dir / "eval.jsonl"
    eval_file = run_dir / "eval.json"
    assert config_file.is_file()
    assert train_file.is_file()
    assert eval_jsonl_file.is_file()
    assert eval_file.is_file()

    config_payload = json.loads(config_file.read_text())
    assert config_payload["experiment"] == "sir_yes_sir"
    assert config_payload["variant"] == "cls"
    assert config_payload["device"] in ("cpu", "mps", "cuda")
    assert "git_commit_sha" in config_payload
    loaded_cfg = SirYesSirClsConfig.from_dict(config_payload["config"])
    assert loaded_cfg.max_steps == 2
    assert loaded_cfg.n_layers == 1

    train_lines = [json.loads(line) for line in train_file.read_text().splitlines()]
    assert len(train_lines) == 2
    assert train_lines[0]["step"] == 1
    assert isinstance(train_lines[0]["loss"], float)
    # `cls` `single_pass` at `L=3`, `batch_size=4` supervises `2 * 3 * 4 = 24` targets per step.
    assert train_lines[0]["tokens_seen"] == 24
    assert train_lines[1]["step"] == 2
    assert isinstance(train_lines[1]["loss"], float)
    assert train_lines[1]["tokens_seen"] == 48

    loaded_eval = json.loads(eval_file.read_text())
    assert loaded_eval == eval_results

    out = capsys.readouterr().out
    assert "[sir_yes_sir:cls] Starting run: B=4, L=3" in out
    assert f"device={config_payload['device']}" in out
    assert "[sir_yes_sir:cls] step=1/2 loss=" in out
    assert "[sir_yes_sir:cls] step=2/2 loss=" in out
    assert "[sir_yes_sir:cls] eval (4 problems):" in out
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
            "--max-steps",
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
            "--d-scratch",
            "0",
        ],
    )
    assert eval_results is not None
    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "sir_yes_sir"
    assert config_payload["variant"] == "ar"
    assert config_payload["device"] in ("cpu", "mps", "cuda")
    loaded_cfg = SirYesSirConfig.from_dict(config_payload["config"])
    assert loaded_cfg.max_steps == 2
    out = capsys.readouterr().out
    assert "[sir_yes_sir:ar] Starting run:" in out


def test_sir_yes_sir_periodic_eval_every_records_history_without_perturbing_training(
    tmp_path: Path,
):
    periodic_dir = tmp_path / "periodic"
    final_only_dir = tmp_path / "final_only"
    common_args = [
        "sir_yes_sir",
        "cls",
        "--max-steps",
        "5",
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
        "--seed",
        "123",
    ]
    res_periodic = run_experiments_cli(
        EXPERIMENTS,
        [*common_args, "--out-dir", str(periodic_dir), "--eval-every", "2"],
    )
    res_final = run_experiments_cli(
        EXPERIMENTS,
        [*common_args, "--out-dir", str(final_only_dir), "--eval-every", "0"],
    )
    assert res_periodic == res_final

    periodic_train = [
        (row["loss"], row["tokens_seen"])
        for row in (
            json.loads(line)
            for line in (periodic_dir / "train.jsonl").read_text().splitlines()
        )
    ]
    final_train = [
        (row["loss"], row["tokens_seen"])
        for row in (
            json.loads(line)
            for line in (final_only_dir / "train.jsonl").read_text().splitlines()
        )
    ]
    assert periodic_train == final_train
    assert [toks for _, toks in periodic_train] == [24, 48, 72, 96, 120]

    eval_rows = [
        json.loads(line)
        for line in (periodic_dir / "eval.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in eval_rows] == [2, 4, 5]
    for row in eval_rows:
        assert isinstance(row["elapsed_sec"], float)
        assert row["splits"]["id"]["exact_match"] == row["exact_match"]
        assert row["splits"]["id"]["step_accuracy"] == row["step_accuracy"]
        assert row["splits"]["id"]["halt_accuracy"] == row["halt_accuracy"]
        assert "wrap_around_accuracy" in row["splits"]["id"]
        wrap_val = row["splits"]["id"]["wrap_around_accuracy"]
        assert wrap_val is None or (0.0 <= wrap_val <= 1.0)
        assert "wrap_around_accuracy" not in row
    assert {
        k: eval_rows[-1][k] for k in ("exact_match", "step_accuracy", "halt_accuracy")
    } == res_periodic
