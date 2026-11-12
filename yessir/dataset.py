"""Shared dataset contract (`LabelledExample` and `Dataset`) across experiments."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from yessir.example import Example, _is_int
from yessir.vocab import VOCAB_SIZE, Token


@dataclass(frozen=True)
class LabelledExample:
    """One model input `Example` paired with its ground-truth `targets`.

    Attributes:
        example: The model input (`input_ids`, optional `token_type_ids`,
            optional `position_ids`).
        targets: Non-empty tuple of target `Token` ids (each in `0..255`).
            For multi-head steps (`MultiTaskHead`), holds one target token per
            head (e.g. `(o_k, w_k, flag_k)`).
    """

    example: Example
    targets: tuple[Token, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example, Example):
            raise TypeError(
                f"example must be an Example, got {type(self.example).__name__}"
            )
        if not isinstance(self.targets, tuple):
            raise TypeError(
                f"targets must be a tuple, got {type(self.targets).__name__}"
            )
        if not self.targets:
            raise ValueError("targets must not be empty")

        for token in self.targets:
            if not _is_int(token) or not (0 <= token < VOCAB_SIZE):
                raise ValueError(
                    f"{token!r} is not a valid target token id (an int in 0..{VOCAB_SIZE - 1})"
                )


class Dataset(Protocol):
    """Protocol implemented by each experiment's data generator."""

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s."""
        ...
