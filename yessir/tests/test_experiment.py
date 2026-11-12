"""Tests for `yessir.experiment` (`ExperimentConfig`, `Experiment`, and CLI config parsing)."""

import json
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Any

import pytest

from yessir.experiment import (
    Experiment,
    ExperimentConfig,
    add_config_args,
    parse_config_cli,
    run_experiment_cli,
    run_experiments_cli,
)


@unique
class _DummyMode(StrEnum):
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass(frozen=True, kw_only=True)
class _DummyConfig(ExperimentConfig):
    steps: int = 100
    lr: float = 1e-3
    mode: _DummyMode = _DummyMode.FAST
    limit: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _DummyMode(self.mode))
        if self.steps < 1:
            raise ValueError("steps must be >= 1")


def test_experiment_config_to_and_from_dict():
    cfg = _DummyConfig(steps=50, mode=_DummyMode.ACCURATE, limit=5, verbose=True)
    as_dict = cfg.to_dict()
    assert as_dict == {
        "steps": 50,
        "lr": 1e-3,
        "mode": "accurate",
        "limit": 5,
        "verbose": True,
    }
    restored = _DummyConfig.from_dict(as_dict)
    assert restored == cfg
    assert isinstance(restored.mode, _DummyMode)


def test_parse_config_cli_defaults_and_overrides():
    default_cfg = parse_config_cli(_DummyConfig, [])
    assert default_cfg == _DummyConfig()

    overridden = parse_config_cli(
        _DummyConfig,
        [
            "--steps",
            "250",
            "--lr",
            "0.005",
            "--mode",
            "accurate",
            "--limit",
            "10",
            "--verbose",
        ],
    )
    assert overridden == _DummyConfig(
        steps=250,
        lr=0.005,
        mode=_DummyMode.ACCURATE,
        limit=10,
        verbose=True,
    )


def test_parse_config_cli_merges_json_file_with_cli_overrides(tmp_path: Path):
    config_file = tmp_path / "base_config.json"
    config_file.write_text(
        json.dumps(
            {
                "steps": 500,
                "lr": 2e-4,
                "mode": "accurate",
                "limit": 12,
                "verbose": True,
            }
        ),
        encoding="utf-8",
    )

    # CLI flags selectively override the JSON values, and --limit none clears an optional int.
    merged = parse_config_cli(
        _DummyConfig,
        [
            "--config",
            str(config_file),
            "--steps",
            "300",
            "--limit",
            "none",
            "--no-verbose",
        ],
    )
    assert merged == _DummyConfig(
        steps=300,
        lr=2e-4,
        mode=_DummyMode.ACCURATE,
        limit=None,
        verbose=False,
    )


def test_parse_config_cli_rejects_non_object_json(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        parse_config_cli(_DummyConfig, ["--config", str(bad_file)])


def test_add_config_args_rejects_non_dataclass():
    import argparse

    class ForgotDataclassDecorator(ExperimentConfig):
        steps: int = 10

    parser = argparse.ArgumentParser()
    with pytest.raises(TypeError, match="must be decorated with @dataclass"):
        add_config_args(parser, ForgotDataclassDecorator)


def test_experiment_descriptor_binding(tmp_path: Path):
    def _dummy_run(cfg: _DummyConfig, out_dir: Path | None) -> dict[str, Any]:
        return {"steps": cfg.steps, "out_dir": str(out_dir)}

    exp = Experiment(
        name="dummy",
        description="Dummy experiment for testing.",
        config_cls=_DummyConfig,
        run=_dummy_run,
    )
    cfg = exp.config_cls(steps=10)
    result = exp.run(cfg, tmp_path)
    assert result == {"steps": 10, "out_dir": str(tmp_path)}


def test_run_experiment_cli_and_run_experiments_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def _dummy_run(cfg: _DummyConfig, out_dir: Path | None) -> dict[str, Any]:
        return {"steps": cfg.steps, "mode": cfg.mode.value, "out_dir": out_dir}

    exp = Experiment(
        name="dummy",
        description="Dummy experiment for testing.",
        config_cls=_DummyConfig,
        run=_dummy_run,
    )

    # Single-experiment CLI runner with explicit --out-dir and --no-save.
    single_res = run_experiment_cli(exp, ["--steps", "12", "--out-dir", str(tmp_path)])
    assert single_res == {"steps": 12, "mode": "fast", "out_dir": tmp_path}

    nosave_res = run_experiment_cli(exp, ["--steps", "12", "--no-save"])
    assert nosave_res == {"steps": 12, "mode": "fast", "out_dir": None}

    # Default out_dir uses results/<experiment>/<date>-<sha>-<adj>-<noun>.
    default_res = run_experiment_cli(exp, ["--steps", "12"])
    assert default_res is not None
    assert default_res["out_dir"].parent.name == "dummy"

    # Multi-experiment CLI dispatcher with no args prints help and returns None.
    assert run_experiments_cli([exp], []) is None
    help_out = capsys.readouterr().out
    assert "dummy" in help_out

    # Multi-experiment CLI dispatcher routes to the named experiment.
    dispatched_res = run_experiments_cli(
        [exp],
        ["dummy", "--steps", "25", "--mode", "accurate", "--out-dir", str(tmp_path)],
    )
    assert dispatched_res == {"steps": 25, "mode": "accurate", "out_dir": tmp_path}

    # Empty or duplicate experiment lists are rejected.
    with pytest.raises(ValueError, match="at least one Experiment"):
        run_experiments_cli([], [])
    with pytest.raises(ValueError, match="duplicate experiment name"):
        run_experiments_cli([exp, exp], [])
