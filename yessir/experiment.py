"""Generic experiment configuration (`ExperimentConfig`), contract (`Experiment`), and CLI parser."""

import argparse
import dataclasses
import json
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self, Union, get_args, get_origin, get_type_hints


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


@dataclass(frozen=True, kw_only=True)
class Experiment[ConfigT: ExperimentConfig]:
    """Descriptor binding an experiment's name, config schema, and execution entry point."""

    name: str
    description: str
    config_cls: type[ConfigT]
    run: Callable[[ConfigT, Path | None], dict[str, Any]]


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
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


def _format_default(value: Any) -> str:
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
        inner_type, is_optional = _unwrap_optional(field_type)
        default_help = f"(default: {_format_default(field.default)})"

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
        raw_json = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_json, dict):
            raise ValueError(
                f"expected a JSON object in {config_path}, got {type(raw_json).__name__}"
            )
        merged.update(raw_json)

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
