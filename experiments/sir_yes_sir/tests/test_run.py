"""Tests for `experiments.sir_yes_sir.run` and the top-level `experiments` CLI dispatcher."""

import pytest

from experiments.__main__ import EXPERIMENTS
from experiments.__main__ import main as experiments_main
from experiments.sir_yes_sir.run import SIR_YES_SIR_EXPERIMENT
from yessir.experiment import run_experiments_cli


def test_sir_yes_sir_registered_in_top_level_experiments():
    names = [exp.name for exp in EXPERIMENTS]
    assert "sir_yes_sir" in names
    assert SIR_YES_SIR_EXPERIMENT in EXPERIMENTS


def test_dispatch_sir_yes_sir_smoke_run(capsys: pytest.CaptureFixture[str]):
    summary = run_experiments_cli(
        EXPERIMENTS,
        [
            "sir_yes_sir",
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

    out = capsys.readouterr().out
    assert "[sir_yes_sir] step=2" in out


def test_experiments_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]):
    experiments_main([])
    out = capsys.readouterr().out
    assert "sir_yes_sir" in out
