"""Alphabets: the ordered symbols one problem uses, chosen from the symbol pool.

An alphabet is a tuple of distinct byte ids from vocab.SYMBOL_RANGE, with at
least 2 symbols. Its order matters: the symbol at position i stands for the
value i.
"""

import random
from collections.abc import Sequence
from enum import Enum

from yessir.vocab import SYMBOL_RANGE


class SymbolSet(bytes, Enum):
    """Named sets of symbols to build alphabets from."""

    ALL = bytes(SYMBOL_RANGE)
    DECIMAL = b"0123456789"
    HEX = b"0123456789ABCDEF"


def create_alphabet(
    symbols: Sequence[int],
    *,
    sample_size: int | None = None,
    rng: random.Random | None = None,
) -> tuple[int, ...]:
    """Create an alphabet from `symbols`.

    - sample_size=None, rng=None:    `symbols` in the order given.
    - sample_size=None, rng=<Random>: all of `symbols`, shuffled.
    - sample_size=k,    rng=<Random>: k distinct symbols drawn uniformly from `symbols`.
    - sample_size=k,    rng=None:    ValueError. Random choices need an explicit
      rng, so that every alphabet can be reproduced from a seed.

    Args:
        symbols: Distinct ints in vocab.SYMBOL_RANGE to build the alphabet from.
        sample_size: Number of symbols to draw, or None to use all of them.
        rng: Source of randomness for shuffling or sampling, or None to keep
            the given order.

    Returns:
        The alphabet: a tuple of at least 2 distinct symbols.

    Raises:
        ValueError: If `symbols` holds a non-symbol or a duplicate, if the
            alphabet would have fewer than 2 symbols or more than `symbols`
            holds, or if `sample_size` is given without `rng`.
    """
    if sample_size is not None and rng is None:
        raise ValueError(
            "an explicit rng (random.Random) is needed when sample_size is given"
        )

    # Check the whole pool first, so the result never depends on which symbols
    # happen to be drawn.
    pool = list(symbols)
    for symbol in pool:
        if not isinstance(symbol, int) or symbol not in SYMBOL_RANGE:
            raise ValueError(f"{symbol!r} is not a symbol (an int in 0x00..0xEF)")
    if len(set(pool)) != len(pool):
        raise ValueError("symbols must not contain duplicates")

    size = len(pool) if sample_size is None else sample_size
    if not 2 <= size <= len(pool):
        raise ValueError(f"alphabet size must be between 2 and {len(pool)}, got {size}")

    if rng is None:
        return tuple(pool)
    return tuple(rng.sample(pool, size))
