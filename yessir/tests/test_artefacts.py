"""Tests for `yessir.artefacts` (`GitInfo`, `generate_run_name`, `default_run_dir`, `write_config_json`, `write_summary_json`)."""

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
    write_config_json,
    write_summary_json,
)
from yessir.experiment import ExperimentConfig


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


def test_config_and_summary_writers(tmp_path: Path):
    run_dir = tmp_path / "2026-11-12-953e4d9-amber-falcon"
    cfg = _SampleConfig(steps=2, lr=5e-4)

    cfg_path = write_config_json(run_dir, cfg)
    assert cfg_path == run_dir / "config.json"
    assert _SampleConfig.from_dict(json.loads(cfg_path.read_text())) == cfg

    fake_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=True,
    )
    summary_path = write_summary_json(
        run_dir,
        experiment_name="sir_yes_sir",
        config=cfg,
        metrics={"final_loss": 0.5, "exact_match": 1.0, "nan_val": math.nan},
        git_info=fake_git,
    )
    assert summary_path == run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary == {
        "experiment": "sir_yes_sir",
        "run_name": "2026-11-12-953e4d9-amber-falcon",
        "git_commit": "953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        "git_short_sha": "953e4d9",
        "git_commit_date": "2026-11-12",
        "git_dirty": True,
        "config": {"steps": 2, "lr": 5e-4},
        "metrics": {"final_loss": 0.5, "exact_match": 1.0, "nan_val": None},
    }
