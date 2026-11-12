"""Run directory naming (`generate_run_name`, `default_run_dir`), git provenance (`GitInfo`), and JSON artefact writers/readers (`write_config_json`, `read_config_dict`, `write_eval_json`, `load_existing_run_configs`)."""

import json
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import petname

from yessir.experiment import ExperimentConfig
from yessir.metrics import json_safe

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, kw_only=True)
class GitInfo:
    """Git commit metadata for run provenance."""

    commit_date: str
    short_sha: str
    commit_sha: str
    dirty: bool


def get_git_info(repo_dir: Path = _REPO_ROOT) -> GitInfo:
    """Read `HEAD` commit date (`YYYY-MM-DD`), short/full SHA, and working-tree dirty state."""

    def _git(*args: str) -> str:
        res = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    try:
        commit_date = _git("show", "-s", "--format=%cs", "HEAD")
        short_sha = _git("rev-parse", "--short=7", "HEAD")
        commit_sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
        return GitInfo(
            commit_date=commit_date,
            short_sha=short_sha,
            commit_sha=commit_sha,
            dirty=dirty,
        )
    except (OSError, subprocess.CalledProcessError):
        return GitInfo(
            commit_date="unknown-date",
            short_sha="nogit",
            commit_sha="nogit",
            dirty=False,
        )


def generate_run_name(
    existing_parent: Path | None = None,
    *,
    git_info: GitInfo | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a unique `<YYYY-MM-DD>-<short_sha>-<adjective>-<noun>` run directory name.

    Uses `random.SystemRandom()` when `rng is None` so that runs sharing the same
    deterministic training `seed` still receive distinct folder names.
    """
    info = git_info if git_info is not None else get_git_info()
    chooser: random.Random = rng if rng is not None else random.SystemRandom()
    prefix = f"{info.commit_date}-{info.short_sha}"

    adjectives: list[str] = list(petname.adjectives)
    nouns: list[str] = list(petname.names)
    for _ in range(len(adjectives) * len(nouns)):
        adj = chooser.choice(adjectives)
        noun = chooser.choice(nouns)
        candidate = f"{prefix}-{adj}-{noun}"
        if existing_parent is None or not (existing_parent / candidate).exists():
            return candidate

    raise RuntimeError(
        f"could not find an unused run name under {existing_parent} with prefix {prefix!r}"
    )


def default_run_dir(
    experiment_name: str,
    *,
    results_root: Path | None = None,
    git_info: GitInfo | None = None,
    rng: random.Random | None = None,
) -> Path:
    """Return a fresh `results/<experiment_name>/<YYYY-MM-DD>-<short_sha>-<adj>-<noun>` path."""
    root = results_root if results_root is not None else (_REPO_ROOT / "results")
    exp_results_dir = root / experiment_name
    run_name = generate_run_name(exp_results_dir, git_info=git_info, rng=rng)
    return exp_results_dir / run_name


def write_config_json(
    out_dir: Path,
    *,
    experiment_name: str,
    variant_name: str,
    config: ExperimentConfig,
    git_info: GitInfo | None = None,
) -> Path:
    """Write experiment/variant names, git provenance, and `config.to_dict()` to `out_dir / 'config.json'`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = git_info if git_info is not None else get_git_info()
    payload: dict[str, Any] = {
        "experiment": experiment_name,
        "variant": variant_name,
        "git_commit_sha": info.commit_sha,
        "git_short_sha": info.short_sha,
        "git_commit_date": info.commit_date,
        "git_dirty": info.dirty,
        "config": config.to_dict(),
    }
    path = out_dir / "config.json"
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_config_dict(path: Path) -> dict[str, Any]:
    """Read a config dictionary from `path` (either a full `config.json` with a `'config'` key or a flat JSON object)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected a JSON object in {path}, got {type(raw).__name__}")
    nested = raw.get("config")
    return nested if isinstance(nested, dict) else raw


def write_eval_json(
    out_dir: Path,
    eval_metrics: Mapping[str, Any],
) -> Path:
    """Write flat final evaluation metrics to `out_dir / 'eval.json'`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval.json"
    path.write_text(
        json.dumps(json_safe(eval_metrics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_existing_run_configs(
    experiment_name: str,
    *,
    variant_name: str | None = None,
    results_root: Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Load `(run_name, config_dict)` for recorded runs under `results/<experiment_name>/`.

    When `variant_name` is given, only runs whose `config.json` records
    `"variant": variant_name` are returned.
    """
    root = results_root if results_root is not None else (_REPO_ROOT / "results")
    exp_results_dir = root / experiment_name
    if not exp_results_dir.is_dir():
        return []

    runs: list[tuple[str, dict[str, Any]]] = []
    for config_path in sorted(exp_results_dir.glob("*/config.json")):
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            if variant_name is not None and raw.get("variant") != variant_name:
                continue
            nested = raw.get("config")
            config_dict = nested if isinstance(nested, dict) else raw
        except (OSError, json.JSONDecodeError):
            continue
        runs.append((config_path.parent.name, config_dict))
    return runs
