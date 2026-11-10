"""Model input construction for the `sir_yes_sir` experiment.

Turns a task state into an `Example`:
1. Concatenates the sections (`[BOS]`, `alphabet`, `[SEP]`, `x`, `[SEP]`, and the
   output section) end-to-end into a single token sequence (`input_ids`).
2. Attaches the coordinate tracks requested by `embedding_mode` (`token_type_ids`
   and/or `position_ids`).
"""

from enum import IntEnum, StrEnum, unique

from yessir.alphabet import create_alphabet
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


@unique
class TokenType(IntEnum):
    """Section identifiers (`token_type_ids`) for `sir_yes_sir` inputs."""

    SPECIAL = 0  # Delimiters and markers ([BOS], [SEP])
    ALPHABET = 1  # Alphabet symbols (s_0 .. s_{B-1})
    INPUT = 2  # Input sequence symbols (x_0 .. x_{L-1})
    OUT = 3  # Output section (step history + [S] in SM; generated trace in AR)


@unique
class EmbeddingMode(StrEnum):
    """Which coordinate side-channels are attached to an `Example`.

    Why there is no `PE`-only mode (`position_ids` without `token_type_ids`) here:
    - Section-resetting `position_ids` (`0..B-1`, `0..L-1`, `0..`) reuse `0, 1, ...`
      in every section, so without `token_type_ids` to identify the section, tokens
      from the alphabet, input `x`, and output section collide at the same coordinates.
    - Global 1D `position_ids` (`0 .. T - 1` across the whole sequence) place `x_k`
      at `B + 2 + k`, so changing the alphabet size `B` or input length `L` shifts
      every downstream position and breaks length/base generalisation compared to `NOPE`.
    """

    NOPE = "nope"  # input_ids only (token_type_ids=None, position_ids=None)
    TOKEN_TYPE_NOPE = (
        "token_type_nope"  # input_ids + token_type_ids (position_ids=None)
    )
    TOKEN_TYPE_PE = "token_type_pe"  # input_ids + token_type_ids + position_ids


def _validate_problem(
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


def _sections_to_example(
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


def build_sm_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    output_history: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the state-machine input `Example` for step `k = len(output_history)`.

    The three pieces of task state (`alphabet`, `x`, `output_history`) are glued
    together end-to-end with `[BOS]`, `[SEP]`, and `[S]` into a single sequence.
    For example, with `alphabet = "0123"`, `x = "130"`, `output_history = "2"` (`k = 1`):

        input_ids:       [BOS]  '0' '1' '2' '3'  [SEP]  '1' '3' '0'  [SEP]  '2'  [S]
        token_type_ids:    0     1   1   1   1     0     2   2   2     0     3    3
        position_ids:      0     0   1   2   3     0     0   1   2     0     0    1

    Notice that in the `OUT` section (`token_type_ids == 3`), `[S]` sits at
    `position_id = k = 1`, aligning it with `x_1 = '3'` in the `INPUT` section.
    """
    alphabet_tokens, x_tokens = _validate_problem(alphabet, x)
    alphabet_set = set(alphabet_tokens)

    out_tokens = tuple(output_history)
    k = len(out_tokens)
    if k >= len(x_tokens):
        raise IndexError(
            f"step index k={k} (from output_history length) out of range "
            f"for input sequence of length {len(x_tokens)}"
        )

    for idx, sym in enumerate(out_tokens):
        if sym not in alphabet_set:
            raise ValueError(
                f"output symbol {sym!r} at position {idx} not found in alphabet"
            )

    sections: list[tuple[tuple[int, ...], TokenType]] = [
        ((SpecialToken.BOS,), TokenType.SPECIAL),
        (alphabet_tokens, TokenType.ALPHABET),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (x_tokens, TokenType.INPUT),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        ((*out_tokens, SpecialToken.STEP), TokenType.OUT),
    ]
    return _sections_to_example(sections, embedding_mode)


def build_ar_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    trace: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the autoregressive (`ar`) input `Example` for `(alphabet, x)` and `trace`.

    The harness is passive: the problem prefix ends at the second `[SEP]`, and any
    `trace` tokens emitted so far follow in a single `TokenType.OUT` section with
    sequential `position_ids = 0, 1, ..., len(trace) - 1`.

    For `alphabet = "0123"`, `x = "130"`, and a complete `OUT_FIRST` trace:

        input_ids:       [BOS] '0' '1' '2' '3' [SEP] '1' '3' '0' [SEP] '2' '0' [CONT] '0' '1' [CONT] '1' '0' [FIN]
        token_type_ids:    0    1   1   1   1    0    2   2   2    0    3   3    3     3   3    3     3   3    3
        position_ids:      0    0   1   2   3    0    0   1   2    0    0   1    2     3   4    5     6   7    8

    At the start of generation (`trace = ()`), the `Example` ends at the second `[SEP]`.
    """
    alphabet_tokens, x_tokens = _validate_problem(alphabet, x)
    trace_tokens = tuple(trace)

    sections: list[tuple[tuple[int, ...], TokenType]] = [
        ((SpecialToken.BOS,), TokenType.SPECIAL),
        (alphabet_tokens, TokenType.ALPHABET),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (x_tokens, TokenType.INPUT),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (trace_tokens, TokenType.OUT),
    ]
    return _sections_to_example(sections, embedding_mode)
