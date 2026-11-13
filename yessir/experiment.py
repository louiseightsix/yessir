"""Generic experiment configuration (`ExperimentConfig`), variant/experiment contracts (`ExperimentVariant`, `Experiment`), and CLI parser."""

import argparse
import dataclasses
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Self, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Base class for validated, JSON-serialisable experiment configuration schemas.

    Subclasses should be `@dataclass(frozen=True, kw_only=True)` and validate
    single-field bounds and cross-parameter invariants in `__post_init__`.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise all dataclass fields to JSON-compatible primitives."""
        result: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            result[field.name] = value.value if isinstance(value, Enum) else value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construct and validate an instance of `cls` from a mapping."""
        return cls(**data)

    def derived_summary(self) -> str:
        """Optional one-line summary of derived quantities for the interactive TUI."""
        return ""


class VariantRunFn[ConfigT: ExperimentConfig](Protocol):
    """Callable contract for an `ExperimentVariant.run` entry point."""

    def __call__(
        self,
        config: ConfigT,
        out_dir: Path | None = None,
        /,
        *,
        device: str = "auto",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, kw_only=True)
class ExperimentVariant[ConfigT: ExperimentConfig]:
    """One execution/harness variant of an `Experiment` (e.g. `'sm_mh'`, `'ar'`)."""

    name: str
    description: str
    config_cls: type[ConfigT]
    run: VariantRunFn[ConfigT]


@dataclass(frozen=True, kw_only=True)
class Experiment:
    """Descriptor grouping all `ExperimentVariant`s that share a task and reference rule."""

    name: str
    description: str
    variants: tuple[ExperimentVariant[Any], ...]

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(
                f"Experiment {self.name!r} must contain at least one ExperimentVariant"
            )
        seen: set[str] = set()
        for variant in self.variants:
            if variant.name in seen:
                raise ValueError(
                    f"duplicate variant name {variant.name!r} in Experiment {self.name!r}"
                )
            seen.add(variant.name)

    def get_variant(self, name: str) -> ExperimentVariant[Any]:
        """Return the variant named `name`, or raise `KeyError`."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        available = ", ".join(v.name for v in self.variants)
        raise KeyError(
            f"unknown variant {name!r} for experiment {self.name!r} (available: {available})"
        )


def unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """If `tp` is `T | None`, return `(T, True)`; otherwise return `(tp, False)`."""
    origin = get_origin(tp)
    if origin in (types.UnionType, Union):
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return non_none[0], True
    return tp, False


def _make_optional_parser(inner_type: Callable[[str], Any]) -> Callable[[str], Any]:
    """Parse `'none'` / `'null'` as `None`, or delegate to `inner_type`."""

    def _parse(raw: str) -> Any:
        if raw.strip().lower() in ("none", "null"):
            return None
        return inner_type(raw)

    _parse.__name__ = getattr(inner_type, "__name__", "value")
    return _parse


def format_config_value(value: Any) -> str:
    """Format a config field value (`None`, `Enum`, or primitive) for CLI/TUI display."""
    if value is None:
        return "None"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def add_config_args(
    parser: argparse.ArgumentParser,
    config_cls: type[ExperimentConfig],
) -> None:
    """Register `--config` and one `--<field-name>` CLI flag per field of `config_cls`.

    Flags use `default=argparse.SUPPRESS` so `build_config_from_args` can tell
    which flags were explicitly passed on the command line (allowing CLI flags
    to selectively override values loaded from `--config <path.json>`).
    """
    if "__dataclass_fields__" not in config_cls.__dict__:
        raise TypeError(f"{config_cls!r} must be decorated with @dataclass")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON config file; explicit CLI flags override its values",
    )

    type_hints = get_type_hints(config_cls)
    for field in dataclasses.fields(config_cls):
        flag = f"--{field.name.replace('_', '-')}"
        field_type = type_hints.get(field.name, field.type)
        inner_type, is_optional = unwrap_optional(field_type)
        default_help = f"(default: {format_config_value(field.default)})"

        if inner_type is bool and not is_optional:
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=f"{field.name} {default_help}",
            )
        elif isinstance(inner_type, type) and issubclass(inner_type, Enum):
            choices_str = ", ".join(str(member.value) for member in inner_type)
            parser.add_argument(
                flag,
                type=(_make_optional_parser(inner_type) if is_optional else inner_type),
                choices=None if is_optional else tuple(inner_type),
                default=argparse.SUPPRESS,
                help=f"{field.name} in {{{choices_str}}} {default_help}",
            )
        else:
            parser.add_argument(
                flag,
                type=(_make_optional_parser(inner_type) if is_optional else inner_type),
                default=argparse.SUPPRESS,
                help=f"{field.name} {default_help}",
            )


def build_config_from_args[ConfigT: ExperimentConfig](
    config_cls: type[ConfigT],
    args: argparse.Namespace,
) -> ConfigT:
    """Build and validate `config_cls` by merging `--config` JSON with explicit CLI flags."""
    merged: dict[str, Any] = {}
    config_path: Path | None = getattr(args, "config", None)
    if config_path is not None:
        from yessir.artefacts import read_config_dict

        merged.update(read_config_dict(config_path))

    arg_dict = vars(args)
    for field in dataclasses.fields(config_cls):
        if field.name in arg_dict:
            merged[field.name] = arg_dict[field.name]

    return config_cls.from_dict(merged)


def parse_config_cli[ConfigT: ExperimentConfig](
    config_cls: type[ConfigT],
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> ConfigT:
    """Parse `argv` into a validated instance of `config_cls`."""
    parser = argparse.ArgumentParser(description=description)
    add_config_args(parser, config_cls)
    args = parser.parse_args(argv)
    return build_config_from_args(config_cls, args)


def _has_explicit_config_args(
    config_cls: type[ExperimentConfig],
    args: argparse.Namespace,
) -> bool:
    """Return `True` if `--config` or any config field flag was explicitly passed."""
    if getattr(args, "config", None) is not None:
        return True
    arg_dict = vars(args)
    return any(field.name in arg_dict for field in dataclasses.fields(config_cls))


def _add_run_control_args(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Attach `--interactive`, `--out-dir`, `--no-save`, and `--device` flags to `parser`."""
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Open the interactive config editor before starting the run",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Directory to write run artefacts into (default: results/<experiment>/<date>-<sha>-<words>)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Do not write run artefacts (config.json, train.jsonl, eval.json) to disk",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS if suppress_defaults else "auto",
        help="Compute device to run on, e.g. auto, cpu, mps, cuda (default: auto)",
    )


def _configure_experiment_parser(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
) -> None:
    """Attach run-control flags and `<variant>` subparsers for `experiment`."""
    _add_run_control_args(parser, suppress_defaults=False)
    subparsers = parser.add_subparsers(
        dest="variant",
        metavar="<variant>",
        help=f"Variant of {experiment.name} to run",
    )
    for variant in experiment.variants:
        v_sub = subparsers.add_parser(
            variant.name,
            help=variant.description,
            description=variant.description,
        )
        _add_run_control_args(v_sub, suppress_defaults=True)
        add_config_args(v_sub, variant.config_cls)


@dataclass(frozen=True, kw_only=True)
class ResolvedExperimentRun:
    """Resolved `(experiment, variant, config, out_dir, device)` from CLI arguments or the interactive TUI."""

    experiment: Experiment
    variant: ExperimentVariant[Any]
    config: ExperimentConfig
    out_dir: Path | None
    device: str = "auto"


def _resolve_experiment(
    experiment: Experiment,
    exp_parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    argv: Sequence[str] | None,
    interactive: bool | None,
    default_interactive_on_tty: bool | None = None,
    results_root: Path | None = None,
    git_info: Any = None,
) -> ResolvedExperimentRun | None:
    tty_default = (
        default_interactive_on_tty
        if default_interactive_on_tty is not None
        else (argv is None)
    ) and sys.stdin.isatty()
    variant_name: str | None = getattr(args, "variant", None)
    if variant_name is None:
        should_edit = (
            interactive
            if interactive is not None
            else (args.interactive or tty_default)
        )
        if not should_edit:
            exp_parser.print_help()
            return None

        from yessir.tui import interactive_edit_config

        edited = interactive_edit_config(experiment)
        if edited is None:
            print("Run cancelled.")
            return None
        chosen_variant, chosen_config = edited
    else:
        chosen_variant = experiment.get_variant(variant_name)
        has_flags = _has_explicit_config_args(chosen_variant.config_cls, args)
        should_edit = (
            interactive
            if interactive is not None
            else (args.interactive or (tty_default and not has_flags))
        )
        chosen_config = build_config_from_args(chosen_variant.config_cls, args)
        if should_edit:
            from yessir.tui import interactive_edit_config

            edited = interactive_edit_config(
                experiment,
                variant=chosen_variant,
                initial=chosen_config,
            )
            if edited is None:
                print("Run cancelled.")
                return None
            chosen_variant, chosen_config = edited

    if args.no_save:
        out_dir: Path | None = None
    elif args.out_dir is not None:
        out_dir = args.out_dir
    else:
        from yessir.artefacts import default_run_dir

        out_dir = default_run_dir(
            experiment.name,
            results_root=results_root,
            git_info=git_info,
        )

    return ResolvedExperimentRun(
        experiment=experiment,
        variant=chosen_variant,
        config=chosen_config,
        out_dir=out_dir,
        device=str(getattr(args, "device", "auto")),
    )


def run_experiment_cli(
    experiment: Experiment,
    argv: Sequence[str] | None = None,
    *,
    interactive: bool | None = None,
) -> dict[str, Any] | None:
    """Parse CLI flags for a single `experiment` and invoke the selected variant's `run`.

    When invoked from a terminal (`argv is None` and `sys.stdin.isatty()`) with no
    explicit config flags—or when `-i` / `--interactive` is passed—opens the
    interactive config editor before starting the run.
    """
    parser = argparse.ArgumentParser(
        prog=experiment.name,
        description=experiment.description,
    )
    _configure_experiment_parser(parser, experiment)
    args = parser.parse_args(argv)
    resolved = _resolve_experiment(
        experiment,
        parser,
        args,
        argv=argv,
        interactive=interactive,
    )
    if resolved is None:
        return None
    return resolved.variant.run(
        resolved.config, resolved.out_dir, device=resolved.device
    )


def resolve_experiments_cli(
    experiments: Sequence[Experiment],
    argv: Sequence[str] | None = None,
    *,
    interactive: bool | None = None,
    default_interactive_on_tty: bool | None = None,
    prog: str = "./yessir.sh run",
    results_root: Path | None = None,
    git_info: Any = None,
) -> ResolvedExperimentRun | None:
    """Resolve `(experiment, variant, config, out_dir)` across registered `Experiment`s."""
    if not experiments:
        raise ValueError("experiments must contain at least one Experiment")

    by_name: dict[str, Experiment] = {}
    exp_parsers: dict[str, argparse.ArgumentParser] = {}
    for exp in experiments:
        if exp.name in by_name:
            raise ValueError(f"duplicate experiment name: {exp.name!r}")
        by_name[exp.name] = exp

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a yessir experiment.",
    )
    subparsers = parser.add_subparsers(
        dest="experiment",
        metavar="<experiment>",
        help="Registered experiment to run",
    )
    for exp in experiments:
        sub = subparsers.add_parser(
            exp.name,
            help=exp.description,
            description=exp.description,
        )
        _configure_experiment_parser(sub, exp)
        exp_parsers[exp.name] = sub

    args = parser.parse_args(argv)
    if args.experiment is None:
        parser.print_help()
        return None

    selected = by_name[args.experiment]
    return _resolve_experiment(
        selected,
        exp_parsers[selected.name],
        args,
        argv=argv,
        interactive=interactive,
        default_interactive_on_tty=default_interactive_on_tty,
        results_root=results_root,
        git_info=git_info,
    )


def run_experiments_cli(
    experiments: Sequence[Experiment],
    argv: Sequence[str] | None = None,
    *,
    interactive: bool | None = None,
) -> dict[str, Any] | None:
    """Top-level CLI dispatcher over a sequence of registered `Experiment`s.

    If invoked with no experiment name, prints available experiments and returns `None`.
    If invoked with an experiment name (and optionally a variant name) and no config
    flags in a terminal (or with `-i`), opens the interactive config editor before
    starting the run.
    """
    resolved = resolve_experiments_cli(
        experiments,
        argv,
        interactive=interactive,
        prog="./yessir.sh run",
    )
    if resolved is None:
        return None
    return resolved.variant.run(
        resolved.config, resolved.out_dir, device=resolved.device
    )
