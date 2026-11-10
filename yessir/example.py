"""Unbatched training and inference input container (`Example`, `SideChannel`, `ChannelType`)."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto

from yessir.vocab import VOCAB_SIZE, Token


class ChannelType(Enum):
    """How a side channel behaves during batch collation and model routing."""

    COORDINATE = auto()  # Non-negative index (e.g. token_type_ids, position_ids); padded with 0
    LABEL = auto()  # Loss target (e.g. labels); padded with -100 (PyTorch ignore_index)
    MASK = auto()  # Binary flag (e.g. loss_mask); padded with 0


@dataclass(frozen=True)
class SideChannel:
    """An auxiliary integer sequence aligned token-for-token with `input_ids`."""

    values: tuple[int, ...]
    channel_type: ChannelType = ChannelType.COORDINATE

    def __post_init__(self) -> None:
        if self.channel_type == ChannelType.COORDINATE and any(v < 0 for v in self.values):
            raise ValueError("COORDINATE side channel values must be non-negative")


@dataclass(frozen=True)
class Example:
    """One unbatched example: a primary token sequence plus optional side channels.

    Guarantees:
    - `input_ids` is non-empty and holds valid byte tokens (0..255).
    - Every side channel in `channels` has the exact same length as `input_ids`.
    - Omitting side channels explicitly denotes NoPE / no auxiliary coordinates.
    """

    input_ids: tuple[Token, ...]
    channels: Mapping[str, SideChannel]

    def __init__(
        self,
        input_ids: tuple[Token, ...],
        channels: Mapping[str, SideChannel] | None = None,
    ) -> None:
        if not input_ids:
            raise ValueError("input_ids must not be empty")

        for token in input_ids:
            if not isinstance(token, int) or not (0 <= token < VOCAB_SIZE):
                raise ValueError(
                    f"{token!r} is not a valid token id (an int in 0..{VOCAB_SIZE - 1})"
                )

        merged: dict[str, SideChannel] = dict(channels) if channels else {}

        length = len(input_ids)
        for name, channel in merged.items():
            if len(channel.values) != length:
                raise ValueError(
                    f"side channel {name!r} length ({len(channel.values)}) "
                    f"must match input_ids length ({length})"
                )

        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "channels", merged)

    def __len__(self) -> int:
        return len(self.input_ids)
