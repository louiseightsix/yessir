"""Shared dataset contract (`IGNORE_INDEX`, `LabelledExample`, and `IterableDataset`) across experiments."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Protocol

from yessir.example import Example, _is_int
from yessir.vocab import VOCAB_SIZE

IGNORE_INDEX: Final[int] = -100


@dataclass(frozen=True)
class LabelledExample:
    """One model input `Example` paired with its ground-truth `labels`.

    Attributes:
        example: The model input (`input_ids`, optional `token_type_ids`,
            optional `position_ids`).
        labels: Non-empty tuple of target `Token` ids (each in `0..255` or
            `IGNORE_INDEX = -100` for masked sequence positions).
            - For multitask steps (`MultiTaskHead`), holds one target token per
              head (e.g. `(o_k, w_k, flag_k)`).
            - For sequence-level next-token prediction (`TiedIdentityHead`), holds
              one target per sequence position (`len(labels) == len(example)`),
              with `IGNORE_INDEX` at positions excluded from the loss.
    """

    example: Example
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example, Example):
            raise TypeError(
                f"example must be an Example, got {type(self.example).__name__}"
            )
        if not isinstance(self.labels, tuple):
            raise TypeError(f"labels must be a tuple, got {type(self.labels).__name__}")
        if not self.labels:
            raise ValueError("labels must not be empty")

        for token in self.labels:
            if not _is_int(token) or (
                token != IGNORE_INDEX and not (0 <= token < VOCAB_SIZE)
            ):
                raise ValueError(
                    f"{token!r} is not a valid label token id (an int in 0..{VOCAB_SIZE - 1} or IGNORE_INDEX={IGNORE_INDEX})"
                )


class IterableDataset(Protocol):
    """Protocol implemented by each experiment's streaming data generator."""

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s."""
        ...
