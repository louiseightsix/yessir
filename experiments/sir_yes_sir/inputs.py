"""Shared model input primitives (`TokenType`, `EmbeddingMode`, `validate_problem`, `sections_to_example`, `resolve_symbol_pool`, `iter_problems`, `add_common_dataset_args`) for `sir_yes_sir`."""

import argparse
import random
from collections.abc import Iterator
from enum import IntEnum, StrEnum, unique

from yessir.alphabet import SymbolSet, create_alphabet
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


def resolve_symbol_pool(
    alphabet_size: int,
    x_length: int,
    symbols: TokenSequence = SymbolSet.ALL,
) -> tuple[int, ...]:
    """Validate `(alphabet_size, x_length, symbols)` and return the symbol pool tuple."""
    symbol_pool = create_alphabet(symbols)
    if not (2 <= alphabet_size <= len(symbol_pool)):
        raise ValueError(
            f"alphabet_size must be in 2..{len(symbol_pool)}, got {alphabet_size}"
        )
    if x_length < 1:
        raise ValueError(f"x_length must be >= 1, got {x_length}")
    return symbol_pool


def iter_problems(
    *,
    alphabet_size: int,
    x_length: int,
    symbol_pool: tuple[int, ...],
    seed: int,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield an infinite deterministic stream of `(alphabet, x)` problem instances from `seed`."""
    rng = random.Random(seed)
    while True:
        alphabet = create_alphabet(symbol_pool, sample_size=alphabet_size, rng=rng)
        x = tuple(rng.choice(alphabet) for _ in range(x_length))
        yield alphabet, x


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Attach `-B`/`--alphabet-size`, `-L`/`--x-length`, `--embedding-mode`, and `--seed` flags."""
    parser.add_argument(
        "-B",
        "--alphabet-size",
        type=int,
        default=4,
        help="Alphabet size B (default: 4)",
    )
    parser.add_argument(
        "-L",
        "--x-length",
        type=int,
        default=3,
        help="Length L of input sequence x (default: 3)",
    )
    parser.add_argument(
        "--embedding-mode",
        type=EmbeddingMode,
        choices=tuple(EmbeddingMode),
        default=EmbeddingMode.TOKEN_TYPE_PE,
        help="Embedding mode for token_type_ids and position_ids (default: token_type_pe)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
