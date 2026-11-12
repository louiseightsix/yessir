"""Tests for `yessir.vertex` (`VertexSettings`, `VertexJobRecord`, `build_custom_job_spec`, `cmd_submit`, `cmd_list`, `cmd_pull`, `cmd_build_image`)."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from yessir.artefacts import GitInfo
from yessir.experiment import Experiment, ExperimentConfig, ExperimentVariant
from yessir.vertex import (
    VertexJobRecord,
    VertexSettings,
    build_custom_job_spec,
    cmd_build_image,
    cmd_list,
    cmd_pull,
    cmd_submit,
    find_local_vertex_runs,
    load_vertex_settings,
    read_vertex_json,
    run_vertex_cli,
    write_vertex_json,
)


@dataclass(frozen=True, kw_only=True)
class _SampleConfig(ExperimentConfig):
    steps: int = 10
    lr: float = 1e-3


def _make_sample_experiment() -> Experiment:
    def _noop_run(cfg: _SampleConfig, out_dir: Path | None) -> dict[str, Any]:
        return {"steps": cfg.steps, "out_dir": str(out_dir)}

    return Experiment(
        name="sir_yes_sir",
        description="Sample experiment.",
        variants=(
            ExperimentVariant(
                name="sm_mh",
                description="State-machine multi-head.",
                config_cls=_SampleConfig,
                run=_noop_run,
            ),
        ),
    )


def _sample_settings() -> VertexSettings:
    return VertexSettings(
        project="test-gcp-project",
        region="us-central1",
        bucket="gs://test-yessir-bucket/",
        image_uri="us-central1-docker.pkg.dev/test-gcp-project/yessir/runner:latest",
    )


def test_load_vertex_settings_valid_and_validation(tmp_path: Path):
    missing_path = tmp_path / "vertex.toml"
    with pytest.raises(FileNotFoundError, match="Copy vertex.example.toml"):
        load_vertex_settings(missing_path)

    missing_path.write_text(
        """
project = "my-proj"
region = "us-central1"
bucket = "gs://my-bucket/"
image_uri = "us-central1-docker.pkg.dev/my-proj/yessir/runner:latest"
""",
        encoding="utf-8",
    )
    settings = load_vertex_settings(missing_path)
    assert settings.project == "my-proj"
    assert settings.bucket == "my-bucket"
    assert settings.machine_type == "g2-standard-4"
    assert settings.accelerator_type == "NVIDIA_L4"
    assert settings.accelerator_count == 1

    # Placeholder rejection
    example_copy = tmp_path / "placeholder.toml"
    example_copy.write_text(
        (Path(__file__).resolve().parents[2] / "vertex.example.toml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder"):
        load_vertex_settings(example_copy)

    # Unknown key rejection
    bad_key = tmp_path / "unknown.toml"
    bad_key.write_text(
        """
project = "my-proj"
region = "us-central1"
bucket = "my-bucket"
image_uri = "us-central1-docker.pkg.dev/my-proj/yessir/runner:latest"
typo_key = 123
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        load_vertex_settings(bad_key)


def test_build_custom_job_spec():
    settings = _sample_settings()
    spec = build_custom_job_spec(
        settings=settings,
        experiment_name="sir_yes_sir",
        variant_name="sm_mh",
        run_name="2026-11-12-953e4d9-amber-falcon",
        git_commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
    )
    pool = spec["workerPoolSpecs"][0]
    assert pool["machineSpec"] == {
        "machineType": "g2-standard-4",
        "acceleratorType": "NVIDIA_L4",
        "acceleratorCount": 1,
    }
    container = pool["containerSpec"]
    assert container["imageUri"] == settings.image_uri
    assert container["env"] == [
        {
            "name": "YESSIR_GIT_SHA",
            "value": "953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        },
        {
            "name": "YESSIR_REPO_URL",
            "value": "https://github.com/louiseightsix/yessir.git",
        },
    ]
    fuse_dir = (
        "/gcs/test-yessir-bucket/results/sir_yes_sir/2026-11-12-953e4d9-amber-falcon"
    )
    assert container["args"] == [
        "sir_yes_sir",
        "sm_mh",
        "--config",
        f"{fuse_dir}/config.json",
        "--out-dir",
        fuse_dir,
    ]


def test_cmd_submit_writes_local_config_and_vertex_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _sample_settings()
    exp = _make_sample_experiment()
    fake_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=False,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:3] == ["git", "branch", "-r"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="  origin/main\n", stderr=""
            )
        if cmd[:3] == ["gcloud", "storage", "cp"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:4] == ["gcloud", "ai", "custom-jobs", "create"]:
            payload = {
                "name": "projects/123/locations/us-central1/customJobs/987654321",
                "createTime": "2026-11-12T23:00:00Z",
                "state": "JOB_STATE_PENDING",
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(payload), stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    record = cmd_submit(
        [exp],
        ["sir_yes_sir", "sm_mh", "--steps", "42", "--lr", "0.005"],
        settings=settings,
        results_root=tmp_path,
        git_info=fake_git,
        interactive=False,
    )
    assert record is not None
    assert record.job_id == "987654321"
    assert record.experiment == "sir_yes_sir"
    assert record.variant == "sm_mh"
    assert record.run_name.startswith("2026-11-12-953e4d9-")

    run_dir = tmp_path / "sir_yes_sir" / record.run_name
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vertex.json").is_file()

    saved_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_cfg["config"] == {"steps": 42, "lr": 0.005}
    assert saved_cfg["git_commit_sha"] == fake_git.commit_sha

    loaded_rec = read_vertex_json(run_dir)
    assert loaded_rec == record
    assert find_local_vertex_runs(tmp_path) == [(run_dir, record)]


def test_cmd_submit_rejects_dirty_or_unpushed_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _sample_settings()
    exp = _make_sample_experiment()
    dirty_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=True,
    )
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        cmd_submit(
            [exp],
            ["sir_yes_sir", "sm_mh", "--steps", "10"],
            settings=settings,
            results_root=tmp_path,
            git_info=dirty_git,
            interactive=False,
        )

    clean_git = GitInfo(
        commit_date="2026-11-12",
        short_sha="953e4d9",
        commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        dirty=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="not on any origin/\\* branch"):
        cmd_submit(
            [exp],
            ["sir_yes_sir", "sm_mh", "--steps", "10"],
            settings=settings,
            results_root=tmp_path,
            git_info=clean_git,
            interactive=False,
        )


def test_cmd_list_and_cmd_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    settings = _sample_settings()
    run_dir = tmp_path / "sir_yes_sir" / "2026-11-12-953e4d9-amber-falcon"
    rec = VertexJobRecord(
        job_name="projects/123/locations/us-central1/customJobs/987654321",
        job_id="987654321",
        display_name="sir_yes_sir-2026-11-12-953e4d9-amber-falcon",
        project=settings.project,
        region=settings.region,
        experiment="sir_yes_sir",
        variant="sm_mh",
        run_name="2026-11-12-953e4d9-amber-falcon",
        git_commit_sha="953e4d9abe8aaa741cb41ecb831b173f59b30f8a",
        repo_url=settings.repo_url,
        image_uri=settings.image_uri,
        gcs_run_uri="gs://test-yessir-bucket/results/sir_yes_sir/2026-11-12-953e4d9-amber-falcon",
        machine_type="g2-standard-4",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        submitted_at="2026-11-12T23:00:00Z",
    )
    write_vertex_json(run_dir, rec)

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:4] == ["gcloud", "ai", "custom-jobs", "list"]:
            jobs = [
                {
                    "name": rec.job_name,
                    "displayName": rec.display_name,
                    "state": "JOB_STATE_SUCCEEDED",
                    "createTime": "2026-11-12T23:00:00Z",
                },
                {
                    "name": "projects/123/locations/us-central1/customJobs/111111111",
                    "displayName": "historical-job",
                    "state": "JOB_STATE_FAILED",
                    "createTime": "2026-11-10T12:00:00Z",
                },
            ]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(jobs), stderr=""
            )
        if cmd[:3] == ["gcloud", "storage", "cp"]:
            (run_dir / "eval.json").write_text(
                '{"exact_match": 1.0}\n', encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    jobs = cmd_list(settings, results_root=tmp_path)
    assert len(jobs) == 2
    list_out = capsys.readouterr().out
    assert "987654321" in list_out
    assert "SUCCEEDED" in list_out
    assert "111111111" in list_out
    assert "historical-job" in list_out

    # Pull populates eval.json, and a subsequent pull skips already-pulled runs unless --force.
    pulled = cmd_pull(results_root=tmp_path)
    assert pulled == [run_dir]
    assert (run_dir / "eval.json").is_file()

    pulled_again = cmd_pull(results_root=tmp_path)
    assert pulled_again == []

    pulled_forced = cmd_pull(run_filter="987654321", force=True, results_root=tmp_path)
    assert pulled_forced == [run_dir]


def test_cmd_build_image_and_cli_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings_file = tmp_path / "vertex.toml"
    settings_file.write_text(
        """
project = "test-gcp-project"
region = "us-central1"
bucket = "test-yessir-bucket"
image_uri = "us-central1-docker.pkg.dev/test-gcp-project/yessir/runner:latest"
""",
        encoding="utf-8",
    )

    recorded: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    settings = load_vertex_settings(settings_file)
    cmd_build_image(settings, local_docker=True, repo_dir=tmp_path)
    assert recorded[-1][:3] == ["docker", "buildx", "build"]

    run_vertex_cli(
        [_make_sample_experiment()],
        ["build-image"],
        settings_path=settings_file,
        results_root=tmp_path,
    )
    assert recorded[-1][:3] == ["gcloud", "builds", "submit"]
