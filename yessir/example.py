"""Single unbatched model inputs (`Example`).

Every example has a primary sequence of byte tokens (`input_ids`) and may also
carry aligned per-token `token_type_ids` and `position_ids`.
"""

from dataclasses import dataclass

from yessir.vocab import (
    MAX_TOKEN_TYPES,
    TOKEN_TYPE_RANGE,
    VOCAB_SIZE,
    Token,
    TokenTypeId,
)


def _is_int(value: object) -> bool:
    """Return True if `value` is an `int` and not a `bool` (`bool` subclasses `int`)."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Example:
    """One unbatched sequence of token ids and its optional `token_type_ids` and `position_ids`.

    Attributes:
        input_ids: Non-empty tuple of byte token ids (each in 0..255).
        token_type_ids: Optional tuple of per-token section ids
            (each in `0..15`, i.e. `TOKEN_TYPE_RANGE`). None when section ids
            are not used.
        position_ids: Optional tuple of per-token position offsets (non-negative
            ints). Unlike Hugging Face—where `position_ids=None` means
            "auto-generate 0, 1, ..., T-1"—here `None` means NoPE (no positional
            ids are given to the model).
    """

    input_ids: tuple[Token, ...]
    token_type_ids: tuple[TokenTypeId, ...] | None = None
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

        if self.token_type_ids is not None:
            if not isinstance(self.token_type_ids, tuple):
                raise TypeError(
                    f"token_type_ids must be a tuple, got {type(self.token_type_ids).__name__}"
                )
            if len(self.token_type_ids) != length:
                raise ValueError(
                    f"token_type_ids length ({len(self.token_type_ids)}) must match input_ids length ({length})"
                )
            for value in self.token_type_ids:
                if not _is_int(value) or value not in TOKEN_TYPE_RANGE:
                    raise ValueError(
                        f"token_type_ids must be ints in 0..{MAX_TOKEN_TYPES - 1}, got {value!r}"
                    )

        if self.position_ids is not None:
            if not isinstance(self.position_ids, tuple):
                raise TypeError(
                    f"position_ids must be a tuple, got {type(self.position_ids).__name__}"
                )
            if len(self.position_ids) != length:
                raise ValueError(
                    f"position_ids length ({len(self.position_ids)}) must match input_ids length ({length})"
                )
            for value in self.position_ids:
                if not _is_int(value) or value < 0:
                    raise ValueError(
                        f"position_ids must contain non-negative ints, got {value!r}"
                    )

    def __len__(self) -> int:
        return len(self.input_ids)
