"""Shared model input primitives (`TokenType`, `EmbeddingMode`, `validate_problem`, `sections_to_example`) for `sir_yes_sir`."""

from enum import IntEnum, StrEnum, unique

from yessir.alphabet import create_alphabet
from yessir.example import Example
from yessir.vocab import TokenSequence


@unique
class TokenType(IntEnum):
    """Section identifiers (`token_type_ids`) for `sir_yes_sir` inputs."""

    SPECIAL = 0  # Delimiters and markers ([BOS], [SEP])
    ALPHABET = 1  # Alphabet symbols (s_0 .. s_{B-1})
    INPUT = 2  # Input sequence symbols (x_0 .. x_{L-1})
    OUTPUT = 3  # Output section (step history + [STEP] in SM; generated trace in AR)


@unique
class EmbeddingMode(StrEnum):
    """Which of `token_type_ids` and `position_ids` are attached to an `Example`.

    Why there is no `PE`-only mode (`position_ids` without `token_type_ids`) here:
    - Section-resetting `position_ids` (`0..B-1`, `0..L-1`, `0..`) reuse `0, 1, ...`
      in every section, so without `token_type_ids` to identify the section, tokens
      from the alphabet, input `x`, and output section collide at the same positions.
    - Global 1D `position_ids` (`0 .. T - 1` across the whole sequence) place `x_k`
      at `B + 2 + k`, so changing the alphabet size `B` or input length `L` shifts
      every downstream position and would likely hurt length/base generalisation
      compared to `NOPE`.
    """

    NOPE = "nope"  # input_ids only (token_type_ids=None, position_ids=None)
    TOKEN_TYPE_NOPE = (
        "token_type_nope"  # input_ids + token_type_ids (position_ids=None)
    )
    TOKEN_TYPE_PE = "token_type_pe"  # input_ids + token_type_ids + position_ids


def validate_problem(
    alphabet: TokenSequence, x: TokenSequence
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate `(alphabet, x)` and return both as tuples of ints."""
    alphabet_tokens = create_alphabet(alphabet)
    alphabet_set = set(alphabet_tokens)

    x_tokens = tuple(x)
    if not x_tokens:
        raise ValueError("input sequence x must contain at least 1 symbol")

    for idx, sym in enumerate(x_tokens):
        if sym not in alphabet_set:
            raise ValueError(
                f"input symbol {sym!r} at position {idx} not found in alphabet"
            )

    return alphabet_tokens, x_tokens


def sections_to_example(
    sections: list[tuple[tuple[int, ...], TokenType]],
    embedding_mode: EmbeddingMode,
) -> Example:
    """Flatten `(tokens, token_type)` sections into an `Example` for `embedding_mode`."""
    input_ids: list[int] = []
    token_type_ids: list[int] = []
    position_ids: list[int] = []
    for tokens, token_type in sections:
        input_ids.extend(tokens)
        token_type_ids.extend([token_type] * len(tokens))
        position_ids.extend(range(len(tokens)))

    if embedding_mode == EmbeddingMode.NOPE:
        return Example(input_ids=tuple(input_ids))
    if embedding_mode == EmbeddingMode.TOKEN_TYPE_NOPE:
        return Example(
            input_ids=tuple(input_ids),
            token_type_ids=tuple(token_type_ids),
        )
    if embedding_mode == EmbeddingMode.TOKEN_TYPE_PE:
        return Example(
            input_ids=tuple(input_ids),
            token_type_ids=tuple(token_type_ids),
            position_ids=tuple(position_ids),
        )
    raise ValueError(f"unknown embedding_mode: {embedding_mode!r}")
