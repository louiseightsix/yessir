"""Tests for `yessir.artefacts` (`GitInfo`, `generate_run_name`, `default_run_dir`, `write_config_json`, `write_eval_json`)."""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from yessir.artefacts import (
    GitInfo,
    default_run_dir,
    generate_run_name,
    get_git_info,
    load_existing_run_params,
    write_config_json,
    write_eval_json,
)
from yessir.experiment import ExperimentConfig, parse_config_cli


@dataclass(frozen=True, kw_only=True)
class _SampleConfig(ExperimentConfig):
    steps: int = 10
    lr: float = 1e-3


def test_get_git_info_in_repo():
    info = get_git_info()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", info.commit_date)
    assert re.fullmatch(r"[0-9a-f]{7}", info.short_sha)
    assert re.fullmatch(r"[0-9a-f]{40}", info.commit_sha)
    assert isinstance(info.dirty, bool)


def test_get_git_info_fallback_outside_repo(tmp_path: Path):
    info = get_git_info(tmp_path)
    assert info == GitInfo(
        commit_date="unknown-date",
        short_sha="nogit",
        commit_sha="nogit",
        dirty=False,
    )


def test_generate_run_name_format_and_collision_avoidance(tmp_path: Path):
    fake_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=False,
    )
    first = generate_run_name(tmp_path, git_info=fake_git, rng=random.Random(123))
    assert re.fullmatch(r"2026-11-12-953e4d9-[a-z]+-[a-z]+", first)

    # Create a directory with `first` and verify the same seeded RNG skips the collision.
    (tmp_path / first).mkdir()
    second = generate_run_name(tmp_path, git_info=fake_git, rng=random.Random(123))
    assert second != first
    assert re.fullmatch(r"2026-11-12-953e4d9-[a-z]+-[a-z]+", second)


def test_default_run_dir_layout(tmp_path: Path):
    fake_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="abcdef0",
        commit_sha="abcdef0123456789abcdef0123456789abcdef01",
        dirty=False,
    )
    run_dir = default_run_dir(
        "sir_yes_sir",
        results_root=tmp_path,
        git_info=fake_git,
        rng=random.Random(7),
    )
    assert run_dir.parent == tmp_path / "sir_yes_sir"
    assert run_dir.name.startswith("2026-11-12-abcdef0-")


def test_config_and_eval_writers(tmp_path: Path):
    run_dir = tmp_path / "2026-11-12-953e4d9-amber-falcon"
    cfg = _SampleConfig(steps=2, lr=5e-4)
    fake_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=True,
    )

    cfg_path = write_config_json(
        run_dir,
        experiment_name="sir_yes_sir",
        config=cfg,
        git_info=fake_git,
    )
    assert cfg_path == run_dir / "config.json"
    cfg_payload = json.loads(cfg_path.read_text())
    assert cfg_payload == {
        "experiment": "sir_yes_sir",
        "git_commit": "953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        "git_short_sha": "953e4d9",
        "git_commit_date": "2026-11-12",
        "git_dirty": True,
        "params": {"steps": 2, "lr": 5e-4},
    }
    # Verify `--config` loads `config.json` directly by reading its `params` dict.
    assert parse_config_cli(_SampleConfig, ["--config", str(cfg_path)]) == cfg

    eval_path = write_eval_json(
        run_dir,
        {"exact_match": 1.0, "step_accuracy": 1.0, "nan_val": math.nan},
    )
    assert eval_path == run_dir / "eval.json"
    eval_payload = json.loads(eval_path.read_text())
    assert eval_payload == {
        "exact_match": 1.0,
        "step_accuracy": 1.0,
        "nan_val": None,
    }


def test_load_existing_run_params(tmp_path: Path):
    # Empty or non-existent experiment directory returns [].
    assert load_existing_run_params("sir_yes_sir", results_root=tmp_path) == []

    exp_dir = tmp_path / "sir_yes_sir"
    cfg1 = _SampleConfig(steps=10, lr=1e-3)
    cfg2 = _SampleConfig(steps=20, lr=2e-3)
    write_config_json(
        exp_dir / "2026-11-12-953e4d9-amber-falcon",
        experiment_name="sir_yes_sir",
        config=cfg1,
    )
    write_config_json(
        exp_dir / "2026-11-12-953e4d9-quiet-otter",
        experiment_name="sir_yes_sir",
        config=cfg2,
    )

    loaded = load_existing_run_params("sir_yes_sir", results_root=tmp_path)
    assert loaded == [
        ("2026-11-12-953e4d9-amber-falcon", {"steps": 10, "lr": 1e-3}),
        ("2026-11-12-953e4d9-quiet-otter", {"steps": 20, "lr": 2e-3}),
    ]
