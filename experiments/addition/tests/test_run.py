"""Tests for `experiments.addition.run`, `train_and_evaluate`, and CLI registration."""

import json
from pathlib import Path

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.addition.config import (
    AdditionAddressedArConfig,
    AdditionAddressedStepArConfig,
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
        "addressed_step_ar",
        "addressed_ar",
    ]


def test_addition_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]):
    addition_main([])
    out = capsys.readouterr().out
    for name in (
        "cls",
        "step_fork",
        "step_ar",
        "ar",
        "addressed_step_ar",
        "addressed_ar",
    ):
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
            "--no-fixed-d-model",
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
    eval_jsonl_file = run_dir / "eval.jsonl"
    eval_file = run_dir / "eval.json"
    eval_splits_file = run_dir / "eval_splits.json"
    assert config_file.is_file()
    assert train_file.is_file()
    assert eval_jsonl_file.is_file()
    assert eval_file.is_file()
    assert eval_splits_file.is_file()

    config_payload = json.loads(config_file.read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "cls"
    loaded_cfg = AdditionClsConfig.from_dict(config_payload["config"])
    assert loaded_cfg.max_steps == 5
    assert loaded_cfg.log_every == 2
    assert loaded_cfg.eval_permuted_canonical is True

    # With max_steps=5 and log_every=2, steps 1, 2, 4, and 5 are logged;
    # each `cls` step at `batch_size=4` supervises `4 * 3 = 12` targets.
    train_lines = [json.loads(line) for line in train_file.read_text().splitlines()]
    assert [row["step"] for row in train_lines] == [1, 2, 4, 5]
    assert [row["tokens_seen"] for row in train_lines] == [12, 24, 48, 60]
    for row in train_lines:
        assert isinstance(row["loss"], float)

    loaded_eval = json.loads(eval_file.read_text())
    assert list(loaded_eval.keys()) == expected_headline_keys
    assert loaded_eval == eval_results

    loaded_splits = json.loads(eval_splits_file.read_text())
    expected_split_metric_keys = {
        "exact_match",
        "step_accuracy",
        "oracle_step_accuracy",
        "digit_accuracy",
        "carry_accuracy",
        "halt_accuracy",
        "address_accuracy",
        "syntax_malformation",
        "out_of_alphabet",
        "prior_intrusion",
        "procedural_error",
    }
    # Top-level splits are always present; multi-base splits (`id`, `extrap_len`)
    # also emit `{split}_base{b}` slices, while `permuted` on `--bases 2,4`
    # excludes B=2 (`permuted.bases == (4,)`), so no `permuted_base*` slice is emitted.
    assert {"id", "ood_base3", "extrap_len", "permuted"}.issubset(loaded_splits.keys())
    assert not any(k.startswith("permuted_base") for k in loaded_splits)
    assert not any(k.startswith("ood_base3_base") for k in loaded_splits)
    assert any(k in loaded_splits for k in ("id_base2", "id_base4"))
    assert any(k in loaded_splits for k in ("extrap_len_base2", "extrap_len_base4"))
    for split_dict in loaded_splits.values():
        assert set(split_dict.keys()) == expected_split_metric_keys
    for split_name, prefix in (
        ("id", ""),
        ("ood_base3", "ood_base3_"),
        ("extrap_len", "extrap_len_"),
        ("permuted", "permuted_"),
    ):
        split_dict = loaded_splits[split_name]
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
            "--no-fixed-d-model",
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
        "permuted_exact_match",
        "permuted_step_accuracy",
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
            "--no-fixed-d-model",
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


def test_dispatch_addition_addressed_step_ar_zero_preamble_smoke_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-calm-heron"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "addressed_step_ar",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "4",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "6",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--no-include-alphabet",
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
            "--no-fixed-d-model",
            "--d-scratch",
            "0",
        ],
    )
    assert eval_results is not None
    assert "exact_match" in eval_results
    assert "ood_base3_exact_match" in eval_results

    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "addressed_step_ar"
    loaded_cfg = AdditionAddressedStepArConfig.from_dict(config_payload["config"])
    assert loaded_cfg.step_order == StepOrder.CARRY_FIRST
    assert loaded_cfg.use_query_tokens is True
    assert loaded_cfg.include_alphabet is False
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "eval_splits.json").is_file()

    out = capsys.readouterr().out
    assert "[addition:addressed_step_ar] Starting run:" in out


def test_dispatch_addition_addressed_ar_zero_preamble_smoke_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "2026-11-16-0000000-vivid-ibis"
    eval_results = run_experiments_cli(
        EXPERIMENTS,
        [
            "addition",
            "addressed_ar",
            "--out-dir",
            str(run_dir),
            "--bases",
            "2,4",
            "--min-length",
            "1",
            "--max-length",
            "4",
            "--eval-ood-bases",
            "3",
            "--eval-extrap-max-length",
            "6",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--no-include-alphabet",
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
            "--d-scratch",
            "0",
        ],
    )
    assert eval_results is not None
    assert "exact_match" in eval_results
    assert "ood_base3_exact_match" in eval_results

    config_payload = json.loads((run_dir / "config.json").read_text())
    assert config_payload["experiment"] == "addition"
    assert config_payload["variant"] == "addressed_ar"
    loaded_cfg = AdditionAddressedArConfig.from_dict(config_payload["config"])
    assert loaded_cfg.step_order == StepOrder.CARRY_FIRST
    assert loaded_cfg.use_query_tokens is True
    assert loaded_cfg.include_alphabet is False
    assert (run_dir / "eval.json").is_file()
    assert (run_dir / "eval_splits.json").is_file()

    out = capsys.readouterr().out
    assert "[addition:addressed_ar] Starting run:" in out


def test_addition_periodic_eval_every_records_eval_jsonl_without_perturbing_training(
    tmp_path: Path,
):
    periodic_dir = tmp_path / "periodic"
    final_only_dir = tmp_path / "final_only"
    common_args = [
        "addition",
        "step_ar",
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
        "5",
        "--log-every",
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
        "--seed",
        "77",
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

    # Training steps, losses, and cumulative `tokens_seen` must be identical regardless of mid-training evaluations.
    periodic_train = [
        (row["step"], row["loss"], row["tokens_seen"])
        for row in (
            json.loads(line)
            for line in (periodic_dir / "train.jsonl").read_text().splitlines()
        )
    ]
    final_train = [
        (row["step"], row["loss"], row["tokens_seen"])
        for row in (
            json.loads(line)
            for line in (final_only_dir / "train.jsonl").read_text().splitlines()
        )
    ]
    assert [step for step, _, _ in periodic_train] == [1, 2, 4, 5]
    assert [toks for _, _, toks in periodic_train] == [6, 12, 24, 30]
    assert periodic_train == final_train
    assert json.loads((periodic_dir / "eval_splits.json").read_text()) == json.loads(
        (final_only_dir / "eval_splits.json").read_text()
    )

    eval_rows = [
        json.loads(line)
        for line in (periodic_dir / "eval.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in eval_rows] == [2, 4, 5]
    headline_keys = [
        "exact_match",
        "step_accuracy",
        "ood_base3_exact_match",
        "ood_base3_step_accuracy",
        "extrap_len_exact_match",
        "extrap_len_step_accuracy",
        "permuted_exact_match",
        "permuted_step_accuracy",
    ]
    for row in eval_rows:
        assert isinstance(row["elapsed_sec"], float)
        for key in headline_keys:
            assert 0.0 <= row[key] <= 1.0
        assert {"id", "ood_base3", "extrap_len", "permuted"}.issubset(
            row["splits"].keys()
        )
        assert any(k in row["splits"] for k in ("id_base2", "id_base4"))
        assert any(k in row["splits"] for k in ("extrap_len_base2", "extrap_len_base4"))
        assert row["splits"]["id"]["exact_match"] == row["exact_match"]
        assert row["splits"]["ood_base3"]["exact_match"] == row["ood_base3_exact_match"]
        assert len(row["splits"]["id"]) == 11

    assert {k: eval_rows[-1][k] for k in headline_keys} == res_periodic
    assert eval_rows[-1]["splits"] == json.loads(
        (periodic_dir / "eval_splits.json").read_text()
    )
