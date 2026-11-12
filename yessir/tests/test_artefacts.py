"""Tests for `yessir.artefacts` (`GitInfo`, `generate_run_name`, `default_run_dir`, `write_config_json`, `write_eval_json`, `Study`, `write_study_json`, `read_study_json`, `load_studies`)."""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from yessir.artefacts import (
    GitInfo,
    Study,
    StudyKind,
    StudyRun,
    StudyStatus,
    default_run_dir,
    generate_run_name,
    get_git_info,
    load_existing_run_configs,
    load_studies,
    read_config_dict,
    read_study_json,
    write_config_json,
    write_eval_json,
    write_study_json,
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
        variant_name="sm_mh",
        config=cfg,
        git_info=fake_git,
    )
    assert cfg_path == run_dir / "config.json"
    cfg_payload = json.loads(cfg_path.read_text())
    assert cfg_payload == {
        "experiment": "sir_yes_sir",
        "variant": "sm_mh",
        "git_commit_sha": "953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        "git_short_sha": "953e4d9",
        "git_commit_date": "2026-11-12",
        "git_dirty": True,
        "config": {"steps": 2, "lr": 5e-4},
    }
    assert read_config_dict(cfg_path) == {"steps": 2, "lr": 5e-4}
    # Verify `--config` loads `config.json` directly by reading its `config` dict.
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


def test_load_existing_run_configs(tmp_path: Path):
    # Empty or non-existent experiment directory returns [].
    assert load_existing_run_configs("sir_yes_sir", results_root=tmp_path) == []

    exp_dir = tmp_path / "sir_yes_sir"
    cfg1 = _SampleConfig(steps=10, lr=1e-3)
    cfg2 = _SampleConfig(steps=20, lr=2e-3)
    write_config_json(
        exp_dir / "2026-11-12-953e4d9-amber-falcon",
        experiment_name="sir_yes_sir",
        variant_name="sm_mh",
        config=cfg1,
    )
    write_config_json(
        exp_dir / "2026-11-12-953e4d9-quiet-otter",
        experiment_name="sir_yes_sir",
        variant_name="ar",
        config=cfg2,
    )

    loaded_all = load_existing_run_configs("sir_yes_sir", results_root=tmp_path)
    assert loaded_all == [
        ("2026-11-12-953e4d9-amber-falcon", {"steps": 10, "lr": 1e-3}),
        ("2026-11-12-953e4d9-quiet-otter", {"steps": 20, "lr": 2e-3}),
    ]

    # Filtering by variant_name only returns runs of that variant.
    loaded_sm_mh = load_existing_run_configs(
        "sir_yes_sir", variant_name="sm_mh", results_root=tmp_path
    )
    assert loaded_sm_mh == [
        ("2026-11-12-953e4d9-amber-falcon", {"steps": 10, "lr": 1e-3}),
    ]
    loaded_ar = load_existing_run_configs(
        "sir_yes_sir", variant_name="ar", results_root=tmp_path
    )
    assert loaded_ar == [
        ("2026-11-12-953e4d9-quiet-otter", {"steps": 20, "lr": 2e-3}),
    ]


def test_study_draft_defaults_and_roundtrip(tmp_path: Path):
    draft = Study(
        title="Carry propagation across longer operands",
        question="Does the model fail on long carry chains before random lengths?",
    )
    assert draft.status == StudyStatus.DRAFT
    assert draft.kind is None
    assert draft.runs == ()
    assert draft.conclusion is None

    path = write_study_json(
        "addition",
        "carry-propagation",
        draft,
        studies_root=tmp_path,
    )
    assert path == tmp_path / "addition" / "carry-propagation.json"
    assert json.loads(path.read_text()) == {
        "title": "Carry propagation across longer operands",
        "status": "draft",
        "kind": None,
        "question": "Does the model fail on long carry chains before random lengths?",
        "runs": [],
        "conclusion": None,
    }
    assert read_study_json(path) == draft


def test_study_with_runs_and_many_to_many_reuse(tmp_path: Path):
    baseline_run = "2026-11-12-a13cc88-closing-filly"
    ablated_run = "2026-11-12-a13cc88-steady-otter"
    fast_lr_run = "2026-11-12-a13cc88-bright-heron"

    ablation_study = Study(
        title="Token-type embedding ablation",
        status=StudyStatus.CONCLUDED,
        kind=StudyKind.ABLATION,
        question="Does disabling use_token_type hurt exact match on sir_yes_sir?",
        runs=[
            StudyRun(run_name=baseline_run, label="baseline (use_token_type=true)"),
            StudyRun(run_name=ablated_run, label="ablated (use_token_type=false)"),
        ],
        conclusion="No degradation on L=4, B=4.",
    )
    sweep_study = Study(
        title="Learning rate sweep",
        status=StudyStatus.IN_PROGRESS,
        kind=StudyKind.SWEEP,
        question="What learning rate converges fastest on L=4, B=4?",
        runs=[
            StudyRun(run_name=baseline_run, label="lr=1e-3"),
            StudyRun(run_name=fast_lr_run, label="lr=3e-3"),
        ],
    )

    write_study_json(
        "sir_yes_sir", "token-type-ablation", ablation_study, studies_root=tmp_path
    )
    write_study_json("sir_yes_sir", "lr-sweep", sweep_study, studies_root=tmp_path)

    loaded = load_studies("sir_yes_sir", studies_root=tmp_path)
    assert loaded == [
        ("lr-sweep", sweep_study),
        ("token-type-ablation", ablation_study),
    ]
    # Non-existent experiment directory returns [].
    assert load_studies("nonexistent", studies_root=tmp_path) == []


def test_study_validation_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="Study.title"):
        Study(title="   ")
    with pytest.raises(ValueError, match="StudyRun.run_name"):
        StudyRun(run_name="")
    with pytest.raises(ValueError, match="study slug"):
        write_study_json(
            "sir_yes_sir",
            "../bad-slug",
            Study(title="Valid title"),
            studies_root=tmp_path,
        )
