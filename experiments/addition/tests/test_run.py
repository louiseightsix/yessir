"""Tests for `experiments.addition.run`, `train_and_evaluate`, and CLI registration."""

import json
from pathlib import Path

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.addition.config import (
    AdditionArConfig,
    AdditionClsConfig,
    AdditionStepArConfig,
    AdditionStepForkConfig,
)
from experiments.addition.inputs import StepOrder
from experiments.addition.run import ADDITION_EXPERIMENT
from experiments.addition.run import main as addition_main
from yessir.experiment import run_experiments_cli


def test_addition_registered_in_top_level_experiments():
    names = [exp.name for exp in EXPERIMENTS]
    assert "addition" in names
    assert ADDITION_EXPERIMENT in EXPERIMENTS
    assert [v.name for v in ADDITION_EXPERIMENT.variants] == [
        "cls",
        "step_fork",
        "step_ar",
        "ar",
    ]


def test_addition_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]):
    addition_main([])
    out = capsys.readouterr().out
    for name in ("cls", "step_fork", "step_ar", "ar"):
        assert name in out


def test_dispatch_addition_cls_smoke_run_writes_headline_and_diagnostic_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-calm-heron"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "cls",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "3",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "5",
            "--eval-permuted-canonical",
            "--max-steps",
            "5",
            "--log-every",
            "2",
            "--batch-size",
            "4",
            "--num-eval-problems",
            "4",
            "--eval-batch-size",
            "2",
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
    expected_headline_keys = [
        "exact_match",
        "step_accuracy",
        "ood_base3_exact_match",
        "ood_base3_step_accuracy",
        "extrap_len_exact_match",
        "extrap_len_step_accuracy",
        "permuted_exact_match",
        "permuted_step_accuracy",
    ]
    # Order matters: `web/` uses the first key (`exact_match`) as `headlineMetric`.
    assert list(eval_results.keys()) == expected_headline_keys
    for val in eval_results.values():
        assert 0.0 <= val <= 1.0

    config_file = run_dir / "config.json"
    train_file = run_dir / "train.jsonl"
    eval_file = run_dir / "eval.json"
    eval_splits_file = run_dir / "eval_splits.json"
    assert config_file.is_file()
    assert train_file.is_file()
    assert eval_file.is_file()
    assert eval_splits_file.is_file()

    config_payload = json.loads(config_file.read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "cls"
    loaded_cfg = AdditionClsConfig.from_dict(config_payload["config"])
    assert loaded_cfg.max_steps == 5
    assert loaded_cfg.log_every == 2
    assert loaded_cfg.eval_permuted_canonical is True

    # With max_steps=5 and log_every=2, steps 1, 2, 4, and 5 are logged.
    train_lines = [json.loads(line) for line in train_file.read_text().splitlines()]
    assert [row["step"] for row in train_lines] == [1, 2, 4, 5]
    for row in train_lines:
        assert isinstance(row["loss"], float)

    loaded_eval = json.loads(eval_file.read_text())
    assert list(loaded_eval.keys()) == expected_headline_keys
    assert loaded_eval == eval_results

    loaded_splits = json.loads(eval_splits_file.read_text())
    assert list(loaded_splits.keys()) == ["id", "ood_base3", "extrap_len", "permuted"]
    expected_split_metric_keys = {
        "exact_match",
        "step_accuracy",
        "digit_accuracy",
        "carry_accuracy",
        "halt_accuracy",
        "syntax_malformation",
        "out_of_alphabet",
        "prior_intrusion",
        "procedural_error",
    }
    for split_name, prefix in (
        ("id", ""),
        ("ood_base3", "ood_base3_"),
        ("extrap_len", "extrap_len_"),
        ("permuted", "permuted_"),
    ):
        split_dict = loaded_splits[split_name]
        assert set(split_dict.keys()) == expected_split_metric_keys
        assert split_dict["exact_match"] == loaded_eval[f"{prefix}exact_match"]
        assert split_dict["step_accuracy"] == loaded_eval[f"{prefix}step_accuracy"]

    out = capsys.readouterr().out
    assert "[addition:cls] Starting run:" in out
    assert "[addition:cls] step=1/5 loss=" in out
    assert "[addition:cls] step=3/5" not in out
    assert "[addition:cls] step=5/5 loss=" in out
    assert "[addition:cls] eval[id] (4 problems):" in out
    assert "[addition:cls] eval[ood_base3] (4 problems):" in out
    assert f"artefacts written to {run_dir}" in out


def test_dispatch_addition_step_fork_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-brisk-finch"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "step_fork",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "2",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "4",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--num-eval-problems",
            "3",
            "--eval-batch-size",
            "2",
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
    assert list(eval_results.keys()) == [
        "exact_match",
        "step_accuracy",
        "ood_base3_exact_match",
        "ood_base3_step_accuracy",
        "extrap_len_exact_match",
        "extrap_len_step_accuracy",
    ]

    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "step_fork"
    loaded_cfg = AdditionStepForkConfig.from_dict(config_payload["config"])
    assert loaded_cfg.batch_size == 2
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "eval_splits.json").is_file()

    out = capsys.readouterr().out
    assert "[addition:step_fork] Starting run:" in out


def test_dispatch_addition_step_ar_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-amber-wren"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "step_ar",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "2",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "4",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--num-eval-problems",
            "3",
            "--eval-batch-size",
            "2",
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
    assert "exact_match" in eval_results

    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "step_ar"
    loaded_cfg = AdditionStepArConfig.from_dict(config_payload["config"])
    assert loaded_cfg.step_order == StepOrder.CARRY_FIRST
    assert loaded_cfg.use_query_tokens is True
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "eval_splits.json").is_file()

    out = capsys.readouterr().out
    assert "[addition:step_ar] Starting run:" in out


def test_dispatch_addition_ar_smoke_run_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-swift-otter"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "ar",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "2",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "4",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--num-eval-problems",
            "3",
            "--eval-batch-size",
            "2",
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
    assert "exact_match" in eval_results

    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "ar"
    loaded_cfg = AdditionArConfig.from_dict(config_payload["config"])
    assert loaded_cfg.step_order == StepOrder.CARRY_FIRST
    assert loaded_cfg.use_query_tokens is True
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "eval_splits.json").is_file()

    out = capsys.readouterr().out
    assert "[addition:ar] Starting run:" in out
