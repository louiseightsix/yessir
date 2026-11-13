"""Shared dataset contract (`IGNORE_INDEX`, `LabelledExample`, and `IterableDataset`) across experiments."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Protocol

from yessir.example import Example, _is_int
from yessir.vocab import VOCAB_SIZE

IGNORE_INDEX: Final[int] = -100


def _validate_label_token(token: object) -> None:
    if not _is_int(token) or (token != IGNORE_INDEX and not (0 <= token < VOCAB_SIZE)):
        raise ValueError(
            f"{token!r} is not a valid label token id (an int in 0..{VOCAB_SIZE - 1} or IGNORE_INDEX={IGNORE_INDEX})"
        )


@dataclass(frozen=True)
class LabelledExample:
    """One model input `Example` paired with its ground-truth `labels`.

    Attributes:
        example: The model input (`input_ids`, optional `token_type_ids`,
            optional `position_ids`).
        labels: Non-empty tuple of target `Token` ids (each in `0..255` or
            `IGNORE_INDEX = -100` for masked sequence positions), or a non-empty
            tuple of per-position head-target tuples:
            - For single-step parallel classification (`ParallelClassificationHead`
              with `lengths`), holds one target token per head (e.g. `(o_k, flag_k)`).
            - For sequence-level parallel classification (`ParallelClassificationHead`
              with `lengths=None`), holds one `H`-tuple of target tokens per sequence
              position (`len(labels) == len(example)`), with `(IGNORE_INDEX,) * H` at
              positions excluded from the loss.
            - For sequence-level next-token prediction (`SubspaceLMHead`), holds
              one pre-shifted target per sequence position (`len(labels) == len(example)`),
              where `labels[t]` is already the target token predicted at position `t`
              (unlike Hugging Face causal LMs, which shift `labels` internally), with
              `IGNORE_INDEX` at positions excluded from the loss.
    """

    example: Example
    labels: tuple[int, ...] | tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example, Example):
            raise TypeError(
                f"example must be an Example, got {type(self.example).__name__}"
            )
        if not isinstance(self.labels, tuple):
            raise TypeError(f"labels must be a tuple, got {type(self.labels).__name__}")
        if not self.labels:
            raise ValueError("labels must not be empty")

        if isinstance(self.labels[0], tuple):
            num_heads = len(self.labels[0])
            if num_heads == 0:
                raise ValueError("per-position label tuples must not be empty")
            for pos_idx, step_labels in enumerate(self.labels):
                if not isinstance(step_labels, tuple):
                    raise TypeError(
                        f"labels[{pos_idx}] must be a tuple, got {type(step_labels).__name__}"
                    )
                if len(step_labels) != num_heads:
                    raise ValueError(
                        f"all per-position label tuples must have {num_heads} elements; "
                        f"labels[{pos_idx}] has {len(step_labels)}"
                    )
                for token in step_labels:
                    _validate_label_token(token)
        else:
            for token in self.labels:
                _validate_label_token(token)


class IterableDataset(Protocol):
    """Protocol implemented by each experiment's streaming data generator."""

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s."""
        ...
