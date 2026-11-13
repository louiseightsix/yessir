"""Run directory naming (`generate_run_name`, `default_run_dir`), git provenance (`GitInfo`), JSON artefact writers/readers (`write_config_json`, `read_config_dict`, `write_eval_json`, `load_existing_run_configs`), and study definitions (`StudyStatus`, `StudyKind`, `StudyRun`, `Study`, `write_study_json`, `read_study_json`, `load_studies`)."""

import json
import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import petname

from yessir.experiment import ExperimentConfig
from yessir.metrics import json_safe

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
        dirty = bool(_git("status", "--porcelain", "--", ".", ":!results"))
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


def _unwrap_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract the nested `'config'` dict if present, otherwise return `raw`."""
    nested = raw.get("config")
    return nested if isinstance(nested, dict) else raw


def read_config_dict(path: Path) -> dict[str, Any]:
    """Read a config dictionary from `path` (either a full `config.json` with a `'config'` key or a flat JSON object)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected a JSON object in {path}, got {type(raw).__name__}")
    return _unwrap_config_dict(raw)


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
            config_dict = _unwrap_config_dict(raw)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append((config_path.parent.name, config_dict))
    return runs


class StudyStatus(StrEnum):
    """Lifecycle stage of a study."""

    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    CONCLUDED = "concluded"


class StudyKind(StrEnum):
    """Experimental design of a study.

    - `exploratory`: baseline, pilot, or one-off run
    - `ablation`: remove a component relative to a baseline
    - `comparison`: A vs B where neither is a removal
    - `sweep`: vary one setting over a range
    """

    EXPLORATORY = "exploratory"
    ABLATION = "ablation"
    COMPARISON = "comparison"
    SWEEP = "sweep"


@dataclass(frozen=True, kw_only=True)
class StudyRun:
    """Reference to a run directory within a study, with an optional label."""

    run_name: str
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str):
            raise TypeError(
                f"StudyRun.run_name must be a str, got {type(self.run_name).__name__}"
            )
        if not self.run_name.strip():
            raise ValueError(
                f"StudyRun.run_name must be a non-empty string, got {self.run_name!r}"
            )
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(
                f"StudyRun.label must be a str or None, got {type(self.label).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {"run_name": self.run_name, "label": self.label}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialise a `StudyRun` from a dictionary."""
        if not isinstance(data, Mapping):
            raise TypeError(
                f"expected a mapping for StudyRun, got {type(data).__name__}"
            )
        run_name_val = data.get("run_name")
        if not isinstance(run_name_val, str):
            raise TypeError(
                f"StudyRun requires a string 'run_name' field, got {run_name_val!r}"
            )
        label_val = data.get("label")
        if label_val is not None and not isinstance(label_val, str):
            raise TypeError(
                f"StudyRun 'label' must be a string or null, got {type(label_val).__name__}"
            )
        return cls(run_name=run_name_val, label=label_val)


# Keep the JSON schema in sync with `StudyFile` / `loadStudies` in `web/src/pages/index.astro`.
@dataclass(frozen=True, kw_only=True)
class Study:
    """A many-to-many grouping of runs around a research question, ablation, comparison, or sweep."""

    title: str
    status: StudyStatus = StudyStatus.DRAFT
    kind: StudyKind | None = None
    question: str | None = None
    runs: Sequence[StudyRun] = ()
    conclusion: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise TypeError(
                f"Study.title must be a str, got {type(self.title).__name__}"
            )
        if not self.title.strip():
            raise ValueError(
                f"Study.title must be a non-empty string, got {self.title!r}"
            )
        object.__setattr__(self, "status", StudyStatus(self.status))
        if self.kind is not None:
            object.__setattr__(self, "kind", StudyKind(self.kind))
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError(
                f"Study.question must be a str or None, got {type(self.question).__name__}"
            )
        if self.conclusion is not None and not isinstance(self.conclusion, str):
            raise TypeError(
                f"Study.conclusion must be a str or None, got {type(self.conclusion).__name__}"
            )
        runs_tuple = tuple(self.runs)
        for item in runs_tuple:
            if not isinstance(item, StudyRun):
                raise TypeError(
                    f"Study.runs elements must be StudyRun instances, got {type(item).__name__}"
                )
        object.__setattr__(self, "runs", runs_tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "title": self.title,
            "status": self.status.value,
            "kind": self.kind.value if self.kind is not None else None,
            "question": self.question,
            "runs": [r.to_dict() for r in self.runs],
            "conclusion": self.conclusion,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialise a `Study` from a dictionary."""
        if not isinstance(data, Mapping):
            raise TypeError(f"expected a mapping for Study, got {type(data).__name__}")
        title = data.get("title")
        if not isinstance(title, str):
            raise TypeError(f"Study requires a string 'title' field, got {title!r}")
        raw_status = data.get("status", StudyStatus.DRAFT.value)
        status = StudyStatus(raw_status)
        raw_kind = data.get("kind")
        kind = StudyKind(raw_kind) if raw_kind is not None else None
        question = data.get("question")
        conclusion = data.get("conclusion")
        raw_runs = data.get("runs", [])
        if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
            raise TypeError("Study 'runs' field must be a list of StudyRun objects")
        runs = tuple(StudyRun.from_dict(item) for item in raw_runs)
        return cls(
            title=title,
            status=status,
            kind=kind,
            question=question,
            runs=runs,
            conclusion=conclusion,
        )


def _validate_study_slug(slug: str) -> None:
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"study slug must match {_SLUG_RE.pattern!r} (lowercase alphanumeric, '-' or '_'), got {slug!r}"
        )


def write_study_json(
    experiment_name: str,
    slug: str,
    study: Study,
    *,
    studies_root: Path | None = None,
) -> Path:
    """Write `study` to `studies/<experiment_name>/<slug>.json`."""
    _validate_study_slug(slug)
    root = studies_root if studies_root is not None else (_REPO_ROOT / "studies")
    exp_studies_dir = root / experiment_name
    exp_studies_dir.mkdir(parents=True, exist_ok=True)
    path = exp_studies_dir / f"{slug}.json"
    path.write_text(
        json.dumps(study.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_study_json(path: Path) -> Study:
    """Read and validate a `Study` from a JSON file at `path`."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected a JSON object in {path}, got {type(raw).__name__}")
    return Study.from_dict(raw)


def load_studies(
    experiment_name: str,
    *,
    studies_root: Path | None = None,
) -> list[tuple[str, Study]]:
    """Load `(slug, Study)` pairs for all `studies/<experiment_name>/*.json` files, sorted by `slug`."""
    root = studies_root if studies_root is not None else (_REPO_ROOT / "studies")
    exp_studies_dir = root / experiment_name
    if not exp_studies_dir.is_dir():
        return []

    studies: list[tuple[str, Study]] = []
    for study_path in sorted(exp_studies_dir.glob("*.json")):
        studies.append((study_path.stem, read_study_json(study_path)))
    return studies
