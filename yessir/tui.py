"""Interactive terminal UI (`ConfigEditorState`, `interactive_edit_config`) for `ExperimentConfig`s.

Separates the pure state machine (`ConfigEditorState`) from terminal I/O so that
arrow-key navigation, value stepping, inline typing, and live cross-parameter
validation can be unit-tested without a TTY.
"""

import dataclasses
import select
import sys
import types
from dataclasses import dataclass
from enum import Enum, StrEnum, unique
from typing import Any, Union, get_args, get_origin, get_type_hints

from yessir.experiment import Experiment, ExperimentConfig


@unique
class EditorOutcome(StrEnum):
    """Result of handling one key event in `ConfigEditorState`."""

    CONTINUE = "continue"
    RUN = "run"
    CANCEL = "cancel"


@dataclass(frozen=True, kw_only=True)
class FieldSpec:
    """Metadata for one editable row in the interactive config table."""

    name: str
    inner_type: Any
    is_optional: bool
    default: Any
    domain: str
    step: int | float
    choices: tuple[Any, ...] | None = None


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin in (types.UnionType, Union):
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return non_none[0], True
    return tp, False


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def build_field_specs(config_cls: type[ExperimentConfig]) -> tuple[FieldSpec, ...]:
    """Introspect `config_cls` fields into `FieldSpec` descriptors for the TUI table."""
    type_hints = get_type_hints(config_cls)
    specs: list[FieldSpec] = []
    for field in dataclasses.fields(config_cls):
        raw_type = type_hints.get(field.name, field.type)
        inner_type, is_optional = _unwrap_optional(raw_type)
        meta = field.metadata

        choices: tuple[Any, ...] | None = None
        if isinstance(inner_type, type) and issubclass(inner_type, Enum):
            choices = tuple(member.value for member in inner_type)
            default_domain = "{" + ", ".join(str(c) for c in choices) + "}"
            if is_optional:
                default_domain += " or None"
        elif inner_type is bool:
            choices = (False, True)
            default_domain = "{False, True}"
        elif inner_type is int:
            default_domain = "int or None" if is_optional else "int"
        elif inner_type is float:
            default_domain = "float or None" if is_optional else "float"
        else:
            default_domain = "str or None" if is_optional else "str"

        domain = str(meta.get("domain", default_domain))
        default_step: int | float = 1e-4 if inner_type is float else 1
        step = meta.get("step", default_step)
        default_val = (
            field.default.value if isinstance(field.default, Enum) else field.default
        )

        specs.append(
            FieldSpec(
                name=field.name,
                inner_type=inner_type,
                is_optional=is_optional,
                default=default_val,
                domain=domain,
                step=step,
                choices=choices,
            )
        )
    return tuple(specs)


def _step_value(spec: FieldSpec, current: Any, delta: int) -> Any:
    """Step `current` left (`delta = -1`) or right (`delta = +1`) along `spec`'s domain."""
    if spec.choices is not None:
        options = (*spec.choices, None) if spec.is_optional else spec.choices
        try:
            idx = options.index(current)
        except ValueError:
            idx = 0
        return options[(idx + delta) % len(options)]

    if spec.inner_type is int:
        step = int(spec.step)
        if current is None:
            base = int(spec.default) if spec.default is not None else 2
            return base if delta > 0 else None
        next_val = int(current) + delta * step
        if spec.is_optional and next_val < 2:
            return None
        min_val = 0 if (isinstance(spec.default, int) and spec.default == 0) else 1
        return max(min_val, next_val)

    if spec.inner_type is float:
        step_f = float(spec.step)
        if current is None:
            base_f = float(spec.default) if spec.default is not None else step_f
            return base_f if delta > 0 else None
        next_f = round(float(current) + delta * step_f, 8)
        return max(step_f, next_f)

    return current


def _parse_inline_value(spec: FieldSpec, text: str) -> Any:
    """Parse an inline-typed string `text` into a value for `spec`."""
    cleaned = text.strip()
    if spec.is_optional and cleaned.lower() in ("", "none", "null"):
        return None

    if spec.inner_type is bool:
        lower = cleaned.lower()
        if lower in ("true", "1", "yes", "y"):
            return True
        if lower in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"expected True or False, got {cleaned!r}")

    if isinstance(spec.inner_type, type) and issubclass(spec.inner_type, Enum):
        for member in spec.inner_type:
            if cleaned.lower() == str(member.value).lower():
                return member.value
        allowed = ", ".join(str(m.value) for m in spec.inner_type)
        raise ValueError(f"expected one of {{{allowed}}}, got {cleaned!r}")

    if spec.inner_type is int:
        return int(cleaned)
    if spec.inner_type is float:
        return float(cleaned)
    return cleaned


class ConfigEditorState[ConfigT: ExperimentConfig]:
    """Pure state machine for interactive arrow-key + inline-typing configuration."""

    def __init__(
        self,
        config_cls: type[ConfigT],
        *,
        title: str = "",
        description: str = "",
        initial: ConfigT | None = None,
    ) -> None:
        self.config_cls = config_cls
        self.title = title or config_cls.__name__
        self.description = description
        self.specs = build_field_specs(config_cls)

        base = initial if initial is not None else config_cls()
        self.values: dict[str, Any] = base.to_dict()
        # Rows 0 .. len(specs) - 1 are parameters; row len(specs) is "[ Run experiment ]".
        self.cursor: int = 0
        self.edit_buffer: str | None = None
        self.message: str | None = None

    @property
    def run_row_index(self) -> int:
        """Row index of the `[ Run experiment ]` action button."""
        return len(self.specs)

    def validate(self) -> tuple[ConfigT | None, str | None]:
        """Validate current `self.values` against `self.config_cls`."""
        try:
            return self.config_cls.from_dict(self.values), None
        except (ValueError, TypeError) as exc:
            return None, str(exc)

    def _commit_inline_edit(self) -> bool:
        """Commit `self.edit_buffer` into `self.values`. Returns `True` on success."""
        if self.edit_buffer is None:
            return True
        spec = self.specs[self.cursor]
        try:
            parsed = _parse_inline_value(spec, self.edit_buffer)
        except ValueError as exc:
            self.message = f"Invalid {spec.name}: {exc}"
            return False
        self.values[spec.name] = parsed
        self.edit_buffer = None
        self.message = None
        return True

    def handle_key(self, key: str) -> EditorOutcome:
        """Process one logical key (`UP`, `DOWN`, `LEFT`, `RIGHT`, `ENTER`, `ESC`, `BACKSPACE`, or char)."""
        total_rows = len(self.specs) + 1

        if self.edit_buffer is not None:
            if key == "ESC":
                self.edit_buffer = None
                self.message = None
                return EditorOutcome.CONTINUE
            if key == "ENTER":
                self._commit_inline_edit()
                return EditorOutcome.CONTINUE
            if key in ("UP", "DOWN"):
                if self._commit_inline_edit():
                    delta = -1 if key == "UP" else 1
                    self.cursor = (self.cursor + delta) % total_rows
                return EditorOutcome.CONTINUE
            if key == "BACKSPACE":
                self.edit_buffer = self.edit_buffer[:-1]
                self.message = None
                return EditorOutcome.CONTINUE
            if len(key) == 1 and key.isprintable():
                self.edit_buffer += key
                self.message = None
            return EditorOutcome.CONTINUE

        # Navigation mode (edit_buffer is None)
        if key in ("ESC", "q", "Q"):
            return EditorOutcome.CANCEL

        if key == "UP":
            self.cursor = (self.cursor - 1) % total_rows
            self.message = None
            return EditorOutcome.CONTINUE

        if key == "DOWN":
            self.cursor = (self.cursor + 1) % total_rows
            self.message = None
            return EditorOutcome.CONTINUE

        if key in ("r", "R"):
            cfg, err = self.validate()
            if cfg is not None:
                return EditorOutcome.RUN
            self.message = f"Cannot run: {err}"
            return EditorOutcome.CONTINUE

        if self.cursor == self.run_row_index:
            if key == "ENTER":
                cfg, err = self.validate()
                if cfg is not None:
                    return EditorOutcome.RUN
                self.message = f"Cannot run: {err}"
            return EditorOutcome.CONTINUE

        spec = self.specs[self.cursor]
        if key in ("LEFT", "RIGHT"):
            delta = -1 if key == "LEFT" else 1
            self.values[spec.name] = _step_value(spec, self.values[spec.name], delta)
            self.message = None
            return EditorOutcome.CONTINUE

        if key == "ENTER":
            self.edit_buffer = _format_value(self.values[spec.name])
            self.message = None
            return EditorOutcome.CONTINUE

        if key == "BACKSPACE":
            self.edit_buffer = ""
            self.message = None
            return EditorOutcome.CONTINUE

        # Typing digits, '-', '.', or letters directly on a row starts inline editing.
        if len(key) == 1 and key.isprintable() and key not in (" ",):
            self.edit_buffer = key
            self.message = None
            return EditorOutcome.CONTINUE

        return EditorOutcome.CONTINUE

    def render_lines(self) -> list[str]:
        """Render the current editor state as plain-text lines."""
        lines: list[str] = [f"=== {self.title} ==="]
        if self.description:
            lines.append(self.description)
        lines.append(
            "Controls: [Up/Down] select row  [Left/Right] adjust value  "
            "[Type/Enter] edit inline  [r] run  [q/Esc] quit"
        )
        lines.append("")
        header = (
            f"  {'Parameter':<20} {'Value':<20} {'Default':<16} {'Allowed / Range'}"
        )
        lines.append(header)
        lines.append("  " + "-" * 78)

        for idx, spec in enumerate(self.specs):
            selected = idx == self.cursor
            prefix = "> " if selected else "  "
            if selected and self.edit_buffer is not None:
                val_str = f"[{self.edit_buffer}_]"
            elif selected:
                val_str = f"< {_format_value(self.values[spec.name])} >"
            else:
                val_str = _format_value(self.values[spec.name])
            def_str = _format_value(spec.default)
            lines.append(
                f"{prefix}{spec.name:<20} {val_str:<20} {def_str:<16} {spec.domain}"
            )

        lines.append("  " + "-" * 78)
        run_prefix = "> " if self.cursor == self.run_row_index else "  "
        lines.append(f"{run_prefix}[ Run experiment ]")
        lines.append("")

        cfg, err = self.validate()
        if cfg is not None:
            extra = cfg.summary_info()
            suffix = f" ({extra})" if extra else ""
            lines.append(f"  Status: VALID{suffix}")
        else:
            lines.append(f"  Status: INVALID - {err}")

        if self.message:
            lines.append(f"  Note:   {self.message}")

        return lines


def _read_key(fd: int) -> str:
    """Read one logical key from raw file descriptor `fd`."""
    import os

    ch = os.read(fd, 1)
    if not ch:
        return "ESC"
    if ch == b"\x03":
        raise KeyboardInterrupt
    if ch in (b"\r", b"\n"):
        return "ENTER"
    if ch in (b"\x7f", b"\x08"):
        return "BACKSPACE"
    if ch == b"\x1b":
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            return "ESC"
        seq = os.read(fd, 2)
        if seq in (b"[A", b"OA"):
            return "UP"
        if seq in (b"[B", b"OB"):
            return "DOWN"
        if seq in (b"[C", b"OC"):
            return "RIGHT"
        if seq in (b"[D", b"OD"):
            return "LEFT"
        return "ESC"
    return ch.decode("utf-8", errors="ignore")


def interactive_edit_config[ConfigT: ExperimentConfig](
    experiment: Experiment[ConfigT],
    initial: ConfigT | None = None,
) -> ConfigT | None:
    """Run the interactive terminal editor for `experiment` and return the chosen config (or `None` if cancelled)."""
    import termios
    import tty

    state = ConfigEditorState(
        experiment.config_cls,
        title=experiment.name,
        description=experiment.description,
        initial=initial,
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Enter alternate screen buffer and hide cursor while in the TUI.
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        tty.setraw(fd)

        while True:
            # Clear screen and move to top-left; use \r\n under raw mode.
            rendered = "\r\n".join(state.render_lines())
            sys.stdout.write("\x1b[2J\x1b[H" + rendered + "\r\n")
            sys.stdout.flush()

            key = _read_key(fd)
            outcome = state.handle_key(key)
            if outcome == EditorOutcome.RUN:
                cfg, _ = state.validate()
                assert cfg is not None
                return cfg
            if outcome == EditorOutcome.CANCEL:
                return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
