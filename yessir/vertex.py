"""Vertex AI Custom Job submission, listing, result syncing, and image building (`./yessir.sh vertex`)."""

import argparse
import dataclasses
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yessir.artefacts import GitInfo, get_git_info, write_config_json
from yessir.experiment import Experiment, resolve_experiments_cli

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTAINER_LOCAL_OUT_DIR = "/workspace/output"
_PLACEHOLDERS = frozenset(
    {
        "your-gcp-project-id",
        "your-gcs-bucket-name",
        "us-central1-docker.pkg.dev/your-gcp-project-id/yessir/runner:latest",
    }
)


@dataclass(frozen=True, kw_only=True)
class VertexSettings:
    """Local GCP / Vertex AI settings loaded from `vertex.toml`."""

    project: str
    region: str
    bucket: str
    image_uri: str
    repo_url: str = "https://github.com/louiseightsix/yessir.git"
    machine_type: str = "g2-standard-4"
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1

    def __post_init__(self) -> None:
        cleaned_bucket = self.bucket.removeprefix("gs://").strip("/")
        object.__setattr__(self, "bucket", cleaned_bucket)
        for field_name in ("project", "region", "bucket", "image_uri", "repo_url"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"vertex.toml: {field_name!r} must be a non-empty string"
                )
            if val.strip() in _PLACEHOLDERS:
                raise ValueError(
                    f"vertex.toml: {field_name!r} is still set to placeholder value {val!r}"
                )
        if self.accelerator_count < 0:
            raise ValueError("vertex.toml: 'accelerator_count' must be >= 0")


def load_vertex_settings(path: Path | None = None) -> VertexSettings:
    """Load and validate `VertexSettings` from `vertex.toml`."""
    settings_path = path if path is not None else (_REPO_ROOT / "vertex.toml")
    if not settings_path.is_file():
        raise FileNotFoundError(
            f"Vertex settings file not found at {settings_path}. "
            f"Copy vertex.example.toml to vertex.toml and fill in your GCP settings."
        )
    raw = tomllib.loads(settings_path.read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(VertexSettings)}
    unknown = set(raw) - known
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in {settings_path}: {unknown_list}")
    missing = {"project", "region", "bucket", "image_uri"} - set(raw)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required keys in {settings_path}: {missing_list}")
    return VertexSettings(**raw)


@dataclass(frozen=True, kw_only=True)
class VertexJobRecord:
    """Metadata written to `results/<experiment>/<run_name>/vertex.json` when a job is submitted."""

    job_name: str
    job_id: str
    display_name: str
    project: str
    region: str
    experiment: str
    variant: str
    run_name: str
    git_commit_sha: str
    repo_url: str
    image_uri: str
    gcs_run_uri: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    submitted_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise job metadata to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VertexJobRecord":
        """Construct a `VertexJobRecord` from a parsed `vertex.json` dictionary."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def write_vertex_json(out_dir: Path, record: VertexJobRecord) -> Path:
    """Write `record` to `out_dir / 'vertex.json'`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vertex.json"
    path.write_text(
        json.dumps(record.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_vertex_json(path: Path) -> VertexJobRecord:
    """Read a `VertexJobRecord` from `path` (`vertex.json` or a run directory containing it)."""
    file_path = path / "vertex.json" if path.is_dir() else path
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected a JSON object in {file_path}")
    return VertexJobRecord.from_dict(raw)


def find_local_vertex_runs(
    results_root: Path | None = None,
) -> list[tuple[Path, VertexJobRecord]]:
    """Return `(run_dir, VertexJobRecord)` for all local run directories containing `vertex.json`."""
    root = results_root if results_root is not None else (_REPO_ROOT / "results")
    if not root.is_dir():
        return []
    found: list[tuple[Path, VertexJobRecord]] = []
    for vertex_path in sorted(root.glob("*/*/vertex.json")):
        try:
            record = read_vertex_json(vertex_path)
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            continue
        found.append((vertex_path.parent, record))
    return found


def build_custom_job_spec(
    *,
    settings: VertexSettings,
    experiment_name: str,
    variant_name: str,
    run_name: str,
    git_commit_sha: str,
    device: str = "auto",
) -> dict[str, Any]:
    """Build the Vertex AI `CustomJobSpec` payload (`workerPoolSpecs`) for `gcloud ai custom-jobs create`.

    Why `--out-dir` points to local container disk (`/workspace/output`) instead of `/gcs/...`:
    `MetricsLogger` opens, appends, and closes `train.jsonl` on every step. On Cloud Storage FUSE,
    each close re-uploads the entire object (O(steps^2) network traffic). Writing to local container
    disk during training and syncing `/workspace/output` to `/gcs/<bucket>/...` on exit via
    `entrypoint.sh` keeps per-step logging O(1).
    """
    fuse_run_dir = f"/gcs/{settings.bucket}/results/{experiment_name}/{run_name}"
    machine_spec: dict[str, Any] = {"machineType": settings.machine_type}
    if settings.accelerator_count > 0 and settings.accelerator_type:
        machine_spec["acceleratorType"] = settings.accelerator_type
        machine_spec["acceleratorCount"] = settings.accelerator_count

    return {
        "workerPoolSpecs": [
            {
                "machineSpec": machine_spec,
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": settings.image_uri,
                    "env": [
                        {"name": "YESSIR_GIT_SHA", "value": git_commit_sha},
                        {"name": "YESSIR_REPO_URL", "value": settings.repo_url},
                        {
                            "name": "YESSIR_LOCAL_OUT_DIR",
                            "value": _CONTAINER_LOCAL_OUT_DIR,
                        },
                        {"name": "YESSIR_GCS_OUT_DIR", "value": fuse_run_dir},
                    ],
                    "args": [
                        experiment_name,
                        variant_name,
                        "--config",
                        f"{fuse_run_dir}/config.json",
                        "--out-dir",
                        _CONTAINER_LOCAL_OUT_DIR,
                        "--device",
                        device,
                    ],
                },
            }
        ]
    }


def is_commit_pushed_to_remote(
    commit_sha: str,
    repo_url: str,
    *,
    repo_dir: Path = _REPO_ROOT,
) -> bool:
    """Return `True` if `commit_sha` is reachable from any branch or tag on `repo_url`.

    Queries `repo_url` directly with `git ls-remote --heads --tags` rather than relying on
    potentially stale local `origin/*` tracking refs or assuming `origin` matches `repo_url`.
    """
    try:
        res = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", repo_url],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    remote_shas: list[str] = []
    for line in res.stdout.splitlines():
        parts = line.strip().split()
        if parts:
            remote_shas.append(parts[0])

    if commit_sha in remote_shas:
        return True

    for remote_sha in remote_shas:
        anc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_sha, remote_sha],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if anc.returncode == 0:
            return True
    return False


def _run_checked(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
) -> str:
    """Run `cmd`, returning `stdout` and surfacing `stderr`/`stdout` in a `RuntimeError` on failure."""
    res = subprocess.run(
        list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip()
        if detail:
            print(detail, file=sys.stderr)
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            f"Command failed (exit {res.returncode}): {cmd_str}"
            + (f"\n{detail}" if detail else "")
        )
    return res.stdout


def cmd_build_image(
    settings: VertexSettings,
    *,
    local_docker: bool = False,
    repo_dir: Path = _REPO_ROOT,
) -> None:
    """Build and push `cloud/vertex/Dockerfile` to `settings.image_uri`."""
    if local_docker:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--push",
            "-t",
            settings.image_uri,
            "-f",
            "cloud/vertex/Dockerfile",
            ".",
        ]
        subprocess.run(cmd, cwd=repo_dir, check=True)
        return

    cloudbuild_config = {
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t",
                    settings.image_uri,
                    "-f",
                    "cloud/vertex/Dockerfile",
                    ".",
                ],
            }
        ],
        "images": [settings.image_uri],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="yessir-cloudbuild-",
        encoding="utf-8",
    ) as tmp:
        json.dump(cloudbuild_config, tmp, indent=2)
        tmp.flush()
        cmd = [
            "gcloud",
            "builds",
            "submit",
            "--project",
            settings.project,
            "--region",
            settings.region,
            "--config",
            tmp.name,
            ".",
        ]
        subprocess.run(cmd, cwd=repo_dir, check=True)


_VERTEX_DEVICES = ("auto", "cpu", "cuda")


def _check_no_disallowed_submit_flags(exp_argv: Sequence[str]) -> None:
    """Reject `--out-dir`, `--no-save`, and unsupported `--device` values on `vertex submit`."""
    for idx, token in enumerate(exp_argv):
        if token == "--no-save":
            raise ValueError("Cannot submit a Vertex job with --no-save")
        if token == "--out-dir" or token.startswith("--out-dir="):
            raise ValueError(
                "--out-dir is not supported with `vertex submit`; runs are always created "
                "under results/<experiment>/<run_name> so `vertex list` and `vertex pull` can track them."
            )
        device_val: str | None = None
        if token == "--device" and idx + 1 < len(exp_argv):
            device_val = exp_argv[idx + 1]
        elif token.startswith("--device="):
            device_val = token.split("=", 1)[1]
        if device_val is not None and device_val.strip().lower() not in _VERTEX_DEVICES:
            choices = ", ".join(repr(d) for d in _VERTEX_DEVICES)
            raise ValueError(
                f"Unsupported --device {device_val!r} for `vertex submit`; expected one of {choices}"
            )


def cmd_submit(
    experiments: Sequence[Experiment],
    argv: Sequence[str],
    *,
    settings: VertexSettings,
    results_root: Path | None = None,
    repo_dir: Path = _REPO_ROOT,
    git_info: GitInfo | None = None,
    interactive: bool | None = None,
    default_interactive_on_tty: bool = False,
) -> VertexJobRecord | None:
    """Resolve experiment config, upload config to GCS, submit a Vertex Custom Job, and record local run files."""
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--machine-type", default=None)
    wrapper.add_argument("--accelerator-type", default=None)
    wrapper.add_argument("--accelerator-count", type=int, default=None)
    wrapper.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow submitting when the local git working tree has uncommitted changes",
    )
    wrapper.add_argument(
        "--allow-unpushed",
        action="store_true",
        help="Allow submitting even if HEAD is not pushed to the remote repo_url",
    )
    vertex_args, exp_argv = wrapper.parse_known_args(argv)
    _check_no_disallowed_submit_flags(exp_argv)

    effective_settings = dataclasses.replace(
        settings,
        machine_type=vertex_args.machine_type or settings.machine_type,
        accelerator_type=(
            vertex_args.accelerator_type
            if vertex_args.accelerator_type is not None
            else settings.accelerator_type
        ),
        accelerator_count=(
            vertex_args.accelerator_count
            if vertex_args.accelerator_count is not None
            else settings.accelerator_count
        ),
    )

    info = git_info if git_info is not None else get_git_info(repo_dir)
    wants_help = not exp_argv or any(a in ("-h", "--help") for a in exp_argv)
    if not wants_help:
        if info.dirty and not vertex_args.allow_dirty:
            raise RuntimeError(
                "Local working tree has uncommitted code changes (outside results/). "
                "Commit and push your changes so the remote Vertex container runs the exact "
                "same code, or pass --allow-dirty."
            )
        if not vertex_args.allow_unpushed and not is_commit_pushed_to_remote(
            info.commit_sha,
            effective_settings.repo_url,
            repo_dir=repo_dir,
        ):
            raise RuntimeError(
                f"Commit {info.short_sha} ({info.commit_sha}) is not pushed to "
                f"{effective_settings.repo_url}. Run `git push` before submitting so the "
                f"Vertex container can fetch it, or pass --allow-unpushed."
            )

    resolved = resolve_experiments_cli(
        experiments,
        exp_argv,
        interactive=interactive,
        default_interactive_on_tty=default_interactive_on_tty,
        prog="./yessir.sh vertex submit",
        results_root=results_root,
        git_info=info,
    )
    if resolved is None or resolved.out_dir is None:
        return None

    out_dir = resolved.out_dir
    run_name = out_dir.name
    display_name = f"{resolved.experiment.name}-{run_name}"
    gcs_run_uri = f"gs://{effective_settings.bucket}/results/{resolved.experiment.name}/{run_name}"

    job_spec = build_custom_job_spec(
        settings=effective_settings,
        experiment_name=resolved.experiment.name,
        variant_name=resolved.variant.name,
        run_name=run_name,
        git_commit_sha=info.commit_sha,
        device=resolved.device,
    )

    with tempfile.TemporaryDirectory(prefix="yessir-vertex-stage-") as stage_str:
        stage_dir = Path(stage_str)
        staged_config = write_config_json(
            stage_dir,
            experiment_name=resolved.experiment.name,
            variant_name=resolved.variant.name,
            config=resolved.config,
            device=resolved.device,
            git_info=info,
        )
        _run_checked(
            [
                "gcloud",
                "storage",
                "cp",
                str(staged_config),
                f"{gcs_run_uri}/config.json",
            ]
        )

        job_spec_path = stage_dir / "job_spec.json"
        job_spec_path.write_text(json.dumps(job_spec, indent=2), encoding="utf-8")
        stdout = _run_checked(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "create",
                "--project",
                effective_settings.project,
                "--region",
                effective_settings.region,
                "--display-name",
                display_name,
                "--config",
                str(job_spec_path),
                "--format=json",
            ]
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_config, out_dir / "config.json")

    created = json.loads(stdout) if stdout.strip() else {}
    job_name = str(created.get("name", ""))
    job_id = job_name.rsplit("/", 1)[-1] if "/" in job_name else job_name
    submitted_at = str(
        created.get("createTime")
        or datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    )

    record = VertexJobRecord(
        job_name=job_name,
        job_id=job_id,
        display_name=display_name,
        project=effective_settings.project,
        region=effective_settings.region,
        experiment=resolved.experiment.name,
        variant=resolved.variant.name,
        run_name=run_name,
        git_commit_sha=info.commit_sha,
        repo_url=effective_settings.repo_url,
        image_uri=effective_settings.image_uri,
        gcs_run_uri=gcs_run_uri,
        machine_type=effective_settings.machine_type,
        accelerator_type=effective_settings.accelerator_type,
        accelerator_count=effective_settings.accelerator_count,
        submitted_at=submitted_at,
    )
    write_vertex_json(out_dir, record)
    print(f"Submitted Vertex job {job_id} ({display_name})")
    print(f"  local dir: {out_dir}")
    print(f"  gcs uri:   {gcs_run_uri}")
    return record


def cmd_list(
    settings: VertexSettings,
    *,
    results_root: Path | None = None,
) -> list[dict[str, Any]]:
    """List all Vertex AI Custom Jobs (including historical ones) paired with local run directories."""
    stdout = _run_checked(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "list",
            "--project",
            settings.project,
            "--region",
            settings.region,
            "--format=json",
        ]
    )
    remote_jobs: list[dict[str, Any]] = json.loads(stdout) if stdout.strip() else []

    local_by_job_name: dict[str, tuple[Path, VertexJobRecord]] = {}
    local_by_display: dict[str, tuple[Path, VertexJobRecord]] = {}
    for run_dir, rec in find_local_vertex_runs(results_root):
        if rec.job_name:
            local_by_job_name[rec.job_name] = (run_dir, rec)
        local_by_display[rec.display_name] = (run_dir, rec)

    rows: list[tuple[str, str, str, str, str, str]] = []
    for job in remote_jobs:
        name = str(job.get("name", ""))
        job_id = name.rsplit("/", 1)[-1] if "/" in name else name
        state = str(job.get("state", "UNKNOWN")).removeprefix("JOB_STATE_")
        created = str(job.get("createTime", ""))[:19].replace("T", " ")
        display = str(job.get("displayName", ""))
        match = local_by_job_name.get(name) or local_by_display.get(display)
        if match is not None:
            run_dir, _ = match
            local_label = f"{run_dir.parent.name}/{run_dir.name}"
            pulled = "yes" if (run_dir / "eval.json").is_file() else "no"
        else:
            local_label = "-"
            pulled = "-"
        rows.append((job_id, state, created, display, local_label, pulled))

    if not rows:
        print("No Vertex AI custom jobs found.")
        return remote_jobs

    headers = ("JOB_ID", "STATE", "CREATED", "DISPLAY_NAME", "LOCAL_RUN", "PULLED")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    return remote_jobs


def cmd_pull(
    *,
    run_filter: str | None = None,
    force: bool = False,
    results_root: Path | None = None,
) -> list[Path]:
    """Pull remote GCS artefacts (`train.jsonl`, `eval.json`, etc.) into local run directories."""
    local_runs = find_local_vertex_runs(results_root)
    targets: list[tuple[Path, VertexJobRecord]] = []
    for run_dir, rec in local_runs:
        if run_filter is not None:
            if run_filter not in (
                rec.run_name,
                rec.display_name,
                rec.job_id,
                f"{rec.experiment}/{rec.run_name}",
            ):
                continue
        elif not force and (run_dir / "eval.json").is_file():
            continue
        targets.append((run_dir, rec))

    if run_filter is not None and not targets:
        raise ValueError(
            f"No local Vertex run matching {run_filter!r} found under results/"
        )

    pulled_dirs: list[Path] = []
    for run_dir, rec in targets:
        print(f"Pulling {rec.gcs_run_uri} -> {run_dir} ...")
        res = subprocess.run(
            ["gcloud", "storage", "cp", f"{rec.gcs_run_uri}/*", str(run_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"  warning: could not pull {rec.run_name}: {res.stderr.strip()}")
            continue
        pulled_dirs.append(run_dir)
    if not targets:
        print("All local Vertex runs already have eval.json (pass --force to re-pull).")
    return pulled_dirs


def run_vertex_cli(
    experiments: Sequence[Experiment],
    argv: Sequence[str] | None = None,
    *,
    settings_path: Path | None = None,
    results_root: Path | None = None,
) -> None:
    """CLI entry point for `./yessir.sh vertex <subcommand>`."""
    from_cli = argv is None
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="./yessir.sh vertex",
        description="Manage Vertex AI Custom Jobs and results for yessir experiments.",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    p_build = sub.add_parser(
        "build-image",
        aliases=["build"],
        help="Build and push the reusable Vertex runner Docker image to Artifact Registry",
    )
    p_build.add_argument(
        "--local-docker",
        action="store_true",
        help="Build and push locally with `docker buildx` instead of Google Cloud Build",
    )

    sub.add_parser(
        "submit",
        help="Create a local run directory, upload config.json, and submit a Vertex AI Custom Job",
        add_help=False,
    )

    sub.add_parser(
        "list",
        help="List all Vertex AI Custom Jobs (including historical jobs) and their local status",
    )

    p_pull = sub.add_parser(
        "pull",
        aliases=["pull-results"],
        help="Pull results from GCS for local Vertex runs",
    )
    p_pull.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Optional run name, job ID, or <experiment>/<run_name> to pull (default: all unpulled runs)",
    )
    p_pull.add_argument(
        "--force",
        action="store_true",
        help="Re-pull runs even if eval.json already exists locally",
    )

    if len(args_list) > 0 and args_list[0] == "submit":
        settings = load_vertex_settings(settings_path)
        cmd_submit(
            experiments,
            args_list[1:],
            settings=settings,
            results_root=results_root,
            default_interactive_on_tty=from_cli,
        )
        return

    parsed = parser.parse_args(args_list)
    if parsed.subcommand is None:
        parser.print_help()
        return

    if parsed.subcommand in ("pull", "pull-results"):
        cmd_pull(
            run_filter=parsed.run,
            force=parsed.force,
            results_root=results_root,
        )
        return

    settings = load_vertex_settings(settings_path)
    if parsed.subcommand in ("build-image", "build"):
        cmd_build_image(settings, local_docker=parsed.local_docker)
    elif parsed.subcommand == "list":
        cmd_list(settings, results_root=results_root)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `python -m yessir.vertex`."""
    from experiments.__main__ import EXPERIMENTS

    run_vertex_cli(EXPERIMENTS, argv)


if __name__ == "__main__":
    main()
