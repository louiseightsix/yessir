"""Tests for the interactive terminal config editor (`yessir.tui`)."""

from dataclasses import dataclass, field
from enum import StrEnum, unique

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.inputs import EmbeddingMode
from yessir.experiment import ExperimentConfig
from yessir.tui import ConfigEditorState, EditorOutcome


@unique
class _Choice(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


@dataclass(frozen=True, kw_only=True)
class _SmallConfig(ExperimentConfig):
    choice: _Choice = _Choice.ALPHA
    count: int = field(default=4, metadata={"domain": "int >= 2", "step": 2})
    rate: float = field(default=0.001, metadata={"domain": "float > 0", "step": 0.0005})
    opt_limit: int | None = field(default=None, metadata={"domain": "int >= 2 or None"})
    flag: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "choice", _Choice(self.choice))
        if self.count < 2:
            raise ValueError("count must be >= 2")
        if self.rate <= 0:
            raise ValueError("rate must be > 0")
        if self.opt_limit is not None and self.opt_limit < 2:
            raise ValueError("opt_limit must be >= 2")


def test_arrow_navigation_and_left_right_value_stepping():
    state = ConfigEditorState(_SmallConfig, title="small", description="Test config")
    assert state.cursor == 0

    # Row 0 is `choice` (Enum): RIGHT cycles alpha -> beta -> gamma -> alpha.
    assert state.values["choice"] == "alpha"
    assert state.handle_key("RIGHT") == EditorOutcome.CONTINUE
    assert state.values["choice"] == "beta"
    assert state.handle_key("RIGHT") == EditorOutcome.CONTINUE
    assert state.values["choice"] == "gamma"
    assert state.handle_key("RIGHT") == EditorOutcome.CONTINUE
    assert state.values["choice"] == "alpha"
    assert state.handle_key("LEFT") == EditorOutcome.CONTINUE
    assert state.values["choice"] == "gamma"

    # DOWN moves to Row 1 (`count`, step=2): RIGHT increments by 2, LEFT decrements by 2.
    state.handle_key("DOWN")
    assert state.cursor == 1
    assert state.values["count"] == 4
    state.handle_key("RIGHT")
    assert state.values["count"] == 6
    state.handle_key("LEFT")
    assert state.values["count"] == 4

    # DOWN moves to Row 2 (`rate`, step=0.0005).
    state.handle_key("DOWN")
    assert state.cursor == 2
    state.handle_key("RIGHT")
    assert state.values["rate"] == 0.0015

    # DOWN moves to Row 3 (`opt_limit`, int | None): RIGHT from None sets to 2; LEFT goes back to None.
    state.handle_key("DOWN")
    assert state.cursor == 3
    assert state.values["opt_limit"] is None
    state.handle_key("RIGHT")
    assert state.values["opt_limit"] == 2
    state.handle_key("LEFT")
    assert state.values["opt_limit"] is None

    # DOWN moves to Row 4 (`flag`, bool): RIGHT toggles False -> True.
    state.handle_key("DOWN")
    assert state.cursor == 4
    assert state.values["flag"] is False
    state.handle_key("RIGHT")
    assert state.values["flag"] is True

    # DOWN moves to the "[ Run experiment ]" row; DOWN again wraps to 0.
    state.handle_key("DOWN")
    assert state.cursor == state.run_row_index
    state.handle_key("DOWN")
    assert state.cursor == 0


def test_inline_typing_commit_and_escape_cancel():
    state = ConfigEditorState(_SmallConfig)
    # Move to `count` (row 1).
    state.handle_key("DOWN")
    assert state.cursor == 1

    # Typing digits directly starts inline editing without needing ENTER first.
    state.handle_key("1")
    state.handle_key("2")
    assert state.edit_buffer == "12"
    state.handle_key("ENTER")
    assert state.edit_buffer is None
    assert state.values["count"] == 12

    # Pressing ENTER opens edit_buffer with current value; ESC cancels without changing value.
    state.handle_key("ENTER")
    assert state.edit_buffer == "12"
    state.handle_key("BACKSPACE")
    state.handle_key("9")
    assert state.edit_buffer == "19"
    state.handle_key("ESC")
    assert state.edit_buffer is None
    assert state.values["count"] == 12

    # Invalid inline text keeps edit_buffer open and records an error note until fixed or cancelled.
    state.handle_key("ENTER")
    state.handle_key("a")
    state.handle_key("ENTER")
    assert state.edit_buffer == "12a"
    assert state.message is not None and "Invalid count" in state.message
    state.handle_key("BACKSPACE")
    state.handle_key("ENTER")
    assert state.edit_buffer is None
    assert state.values["count"] == 12


def test_live_cross_parameter_validation_and_run_blocking():
    state = ConfigEditorState(SirYesSirConfig, title="sir_yes_sir")
    cfg, err = state.validate()
    assert cfg is not None and err is None

    # Find `embedding_mode` row and step it from `token_type_pe` to `nope` while `d_position` is still 32.
    emb_idx = next(i for i, s in enumerate(state.specs) if s.name == "embedding_mode")
    state.cursor = emb_idx
    state.handle_key("RIGHT")  # token_type_pe -> nope
    assert state.values["embedding_mode"] == EmbeddingMode.NOPE

    # Now cross-parameter validation fails because d_position == 32 != 0.
    cfg, err = state.validate()
    assert cfg is None
    assert err is not None and "d_position must be 0" in err

    # Pressing 'r' while invalid is blocked and sets a helpful message.
    assert state.handle_key("r") == EditorOutcome.CONTINUE
    assert state.message is not None and "Cannot run" in state.message
    rendered = "\n".join(state.render_lines())
    assert "Status: INVALID" in rendered

    # Navigate to `d_position` and type 0 inline; status becomes VALID again!
    d_pos_idx = next(i for i, s in enumerate(state.specs) if s.name == "d_position")
    state.cursor = d_pos_idx
    state.handle_key("0")
    state.handle_key("ENTER")
    assert state.values["d_position"] == 0

    cfg, err = state.validate()
    assert cfg is not None and err is None
    rendered_valid = "\n".join(state.render_lines())
    assert "Status: VALID (d_model=464, active_symbols=240)" in rendered_valid

    # Now pressing 'r' (or ENTER on the Run row) succeeds and returns RUN.
    assert state.handle_key("r") == EditorOutcome.RUN


def test_quit_cancels_editor():
    state = ConfigEditorState(_SmallConfig)
    assert state.handle_key("q") == EditorOutcome.CANCEL
    assert state.handle_key("ESC") == EditorOutcome.CANCEL


def test_duplicate_run_alert_and_confirmation():
    default_params = _SmallConfig().to_dict()
    state = ConfigEditorState(
        _SmallConfig,
        existing_runs=[("2026-11-12-953e4d9-amber-falcon", default_params)],
    )

    # Matching run is detected immediately and displayed in rendered lines.
    assert state.matching_runs() == ["2026-11-12-953e4d9-amber-falcon"]
    rendered = "\n".join(state.render_lines())
    assert (
        "Alert:  Parameters match existing run(s): 2026-11-12-953e4d9-amber-falcon"
        in rendered
    )

    # First press of 'r' pauses to ask for confirmation instead of immediately running.
    assert state.handle_key("r") == EditorOutcome.CONTINUE
    assert state.confirm_duplicate is True
    assert state.message is not None and "Duplicate of" in state.message

    # Changing a parameter clears the duplicate alert and confirmation state.
    state.handle_key("RIGHT")
    assert state.matching_runs() == []
    assert state.confirm_duplicate is False

    # Stepping back to the matching parameter value re-triggers the alert,
    # and pressing 'r' twice confirms and runs.
    state.handle_key("LEFT")
    assert state.matching_runs() == ["2026-11-12-953e4d9-amber-falcon"]
    assert state.handle_key("r") == EditorOutcome.CONTINUE
    assert state.handle_key("r") == EditorOutcome.RUN
