"""Single unbatched model inputs (`Example`).

Every example has a primary sequence of byte tokens (`input_ids`) and may also
carry aligned per-token coordinate side channels (`token_type_ids`, `position_ids`).
"""

from dataclasses import dataclass

from yessir.vocab import VOCAB_SIZE, Token


def _is_int(value: object) -> bool:
    """Return True if `value` is an `int` and not a `bool` (`bool` subclasses `int`)."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Example:
    """One unbatched sequence of token ids and its optional side-channel coordinates.

    Attributes:
        input_ids: Non-empty tuple of byte token ids (each in 0..255).
        token_type_ids: Optional tuple of per-token section / role ids
            (non-negative ints, like BERT's segment ids). None when section
            roles are not used.
        position_ids: Optional tuple of per-token position offsets (non-negative
            ints). Unlike Hugging Face—where `position_ids=None` means
            "auto-generate 0, 1, ..., T-1"—here `None` means NoPE (no positional
            coordinates are given to the model).
    """

    input_ids: tuple[Token, ...]
    token_type_ids: tuple[int, ...] | None = None
    position_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, tuple):
            raise TypeError(
                f"input_ids must be a tuple, got {type(self.input_ids).__name__}"
            )
        if not self.input_ids:
            raise ValueError("input_ids must not be empty")

        for token in self.input_ids:
            if not _is_int(token) or not (0 <= token < VOCAB_SIZE):
                raise ValueError(
                    f"{token!r} is not a valid token id (an int in 0..{VOCAB_SIZE - 1})"
                )

        length = len(self.input_ids)
        for name, coords in (
            ("token_type_ids", self.token_type_ids),
            ("position_ids", self.position_ids),
        ):
            if coords is None:
                continue
            if not isinstance(coords, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(coords).__name__}")
            if len(coords) != length:
                raise ValueError(
                    f"{name} length ({len(coords)}) must match input_ids length ({length})"
                )
            for value in coords:
                if not _is_int(value) or value < 0:
                    raise ValueError(
                        f"{name} must contain non-negative ints, got {value!r}"
                    )

    def __len__(self) -> int:
        return len(self.input_ids)
