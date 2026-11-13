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
    ExperimentVariant,
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
    with pytest.raises(TypeError, match="expected a JSON object"):
        parse_config_cli(_DummyConfig, ["--config", str(bad_file)])


def test_add_config_args_rejects_non_dataclass():
    import argparse

    class ForgotDataclassDecorator(ExperimentConfig):
        steps: int = 10

    parser = argparse.ArgumentParser()
    with pytest.raises(TypeError, match="must be decorated with @dataclass"):
        add_config_args(parser, ForgotDataclassDecorator)


@dataclass(frozen=True, kw_only=True)
class _AltDummyConfig(_DummyConfig):
    extra_flag: str = "out_first"


def test_experiment_and_variant_descriptor_binding(tmp_path: Path):
    def _dummy_run(
        cfg: _DummyConfig, out_dir: Path | None = None, *, device: str = "auto"
    ) -> dict[str, Any]:
        return {
            "variant": "v1",
            "steps": cfg.steps,
            "out_dir": str(out_dir),
            "device": device,
        }

    def _alt_run(
        cfg: _AltDummyConfig, out_dir: Path | None = None, *, device: str = "auto"
    ) -> dict[str, Any]:
        return {
            "variant": "v2",
            "steps": cfg.steps,
            "extra_flag": cfg.extra_flag,
            "out_dir": str(out_dir),
            "device": device,
        }

    v1 = ExperimentVariant(
        name="v1",
        description="First variant.",
        config_cls=_DummyConfig,
        run=_dummy_run,
    )
    v2 = ExperimentVariant(
        name="v2",
        description="Second variant with extra flag.",
        config_cls=_AltDummyConfig,
        run=_alt_run,
    )
    exp = Experiment(
        name="dummy",
        description="Dummy experiment for testing.",
        variants=(v1, v2),
    )

    assert exp.get_variant("v1") is v1
    assert exp.get_variant("v2") is v2
    with pytest.raises(KeyError, match="unknown variant 'v3'"):
        exp.get_variant("v3")

    cfg = v1.config_cls(steps=10)
    result = v1.run(cfg, tmp_path)
    assert result == {
        "variant": "v1",
        "steps": 10,
        "out_dir": str(tmp_path),
        "device": "auto",
    }

    with pytest.raises(ValueError, match="at least one ExperimentVariant"):
        Experiment(name="empty", description="Empty.", variants=())
    with pytest.raises(ValueError, match="duplicate variant name 'v1'"):
        Experiment(name="dup", description="Dup.", variants=(v1, v1))


def test_run_experiment_cli_and_run_experiments_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def _v1_run(
        cfg: _DummyConfig, out_dir: Path | None = None, *, device: str = "auto"
    ) -> dict[str, Any]:
        return {
            "variant": "cls",
            "steps": cfg.steps,
            "mode": cfg.mode.value,
            "out_dir": out_dir,
            "device": device,
        }

    def _v2_run(
        cfg: _AltDummyConfig, out_dir: Path | None = None, *, device: str = "auto"
    ) -> dict[str, Any]:
        return {
            "variant": "ar",
            "steps": cfg.steps,
            "extra_flag": cfg.extra_flag,
            "out_dir": out_dir,
            "device": device,
        }

    v_cls = ExperimentVariant(
        name="cls",
        description="Parallel classification variant.",
        config_cls=_DummyConfig,
        run=_v1_run,
    )
    v_ar = ExperimentVariant(
        name="ar",
        description="Autoregressive variant.",
        config_cls=_AltDummyConfig,
        run=_v2_run,
    )
    exp = Experiment(
        name="dummy",
        description="Dummy experiment for testing.",
        variants=(v_cls, v_ar),
    )

    # Single-experiment CLI runner with explicit variant, --out-dir, and --no-save.
    single_res = run_experiment_cli(
        exp, ["cls", "--steps", "12", "--out-dir", str(tmp_path)]
    )
    assert single_res == {
        "variant": "cls",
        "steps": 12,
        "mode": "fast",
        "out_dir": tmp_path,
        "device": "auto",
    }

    nosave_res = run_experiment_cli(exp, ["cls", "--steps", "12", "--no-save"])
    assert nosave_res == {
        "variant": "cls",
        "steps": 12,
        "mode": "fast",
        "out_dir": None,
        "device": "auto",
    }

    # Default out_dir uses results/<experiment>/<date>-<sha>-<adj>-<noun>.
    default_res = run_experiment_cli(exp, ["cls", "--steps", "12"])
    assert default_res is not None
    assert default_res["out_dir"].parent.name == "dummy"

    # Invoking single-experiment CLI with no variant in non-interactive mode prints help.
    assert run_experiment_cli(exp, [], interactive=False) is None
    exp_help = capsys.readouterr().out
    assert "cls" in exp_help and "ar" in exp_help

    # Multi-experiment CLI dispatcher with no args prints help and returns None.
    assert run_experiments_cli([exp], []) is None
    help_out = capsys.readouterr().out
    assert "dummy" in help_out

    # Multi-experiment CLI dispatcher with experiment but no variant in non-interactive mode prints experiment help.
    assert run_experiments_cli([exp], ["dummy"], interactive=False) is None
    sub_help = capsys.readouterr().out
    assert "cls" in sub_help and "ar" in sub_help

    # Multi-experiment CLI dispatcher routes to the named experiment and variant.
    dispatched_cls = run_experiments_cli(
        [exp],
        [
            "dummy",
            "cls",
            "--steps",
            "25",
            "--mode",
            "accurate",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert dispatched_cls == {
        "variant": "cls",
        "steps": 25,
        "mode": "accurate",
        "out_dir": tmp_path,
        "device": "auto",
    }

    dispatched_ar = run_experiments_cli(
        [exp],
        [
            "dummy",
            "ar",
            "--steps",
            "30",
            "--extra-flag",
            "wrap_first",
            "--device",
            "mps",
            "--no-save",
        ],
    )
    assert dispatched_ar == {
        "variant": "ar",
        "steps": 30,
        "extra_flag": "wrap_first",
        "out_dir": None,
        "device": "mps",
    }

    # Empty or duplicate experiment lists are rejected.
    with pytest.raises(ValueError, match="at least one Experiment"):
        run_experiments_cli([], [])
    with pytest.raises(ValueError, match="duplicate experiment name"):
        run_experiments_cli([exp, exp], [])
