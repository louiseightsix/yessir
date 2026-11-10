"""Byte-level vocabulary definitions, token sets, and alphabet sampling."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated

# Semantic token types (byte tokens in 0..255)
type Token = Annotated[int, "Value in 0..255"]
type TokenSequence = Sequence[Token] | bytes

# ---------------------------------------------------------------------------
# 1. The vocabulary and how it is split
# ---------------------------------------------------------------------------
#
# Every token is one byte, so the model's vocabulary is the 256 byte values.
# They are split into contiguous ranges:
#
#   0x00                     0xEF  0xF0            0xF7  0xF8            0xFF
#  ┌─────────────────────────────┬─────────────────────┬─────────────────────┐
#  │         Symbol pool         │   Shared special    │ Experiment special  │
#  │         (240 bytes)         │      (8 bytes)      │      (8 bytes)      │
#  └─────────────────────────────┴─────────────────────┴─────────────────────┘
#                                └────────────── Special tokens ─────────────┘
#
# Alphabets are drawn from the symbol pool. Special tokens (BOS, PAD, ...) have
# fixed ids. The shared range holds the ones every experiment uses; the
# experiment range is reserved for ones that only a single experiment needs.

VOCAB_SIZE = 256

SYMBOL_RANGE = range(0x00, 0xF0)  # noqa: PIE808 (explicit start, to match lines below)
SPECIAL_RANGE = range(0xF0, 0x100)
SHARED_SPECIAL_RANGE = range(0xF0, 0xF8)
EXPERIMENT_SPECIAL_RANGE = range(0xF8, 0x100)

assert (
    len(SYMBOL_RANGE) + len(SHARED_SPECIAL_RANGE) + len(EXPERIMENT_SPECIAL_RANGE)
    == VOCAB_SIZE
), "Vocabulary partition ranges must exactly cover VOCAB_SIZE"


# ---------------------------------------------------------------------------
# 2. Token Enums
# ---------------------------------------------------------------------------


class SymbolSets(bytes, Enum):
    """Predefined collections of symbols for alphabet selection."""

    ALL = bytes(SYMBOL_RANGE)
    DECIMAL = b"0123456789"
    HEX = b"0123456789ABCDEF"


# Static validation: ensure every predefined symbol set contains only valid symbols
for _symbol_set in SymbolSets:
    for _byte in _symbol_set.value:
        assert _byte in SYMBOL_RANGE, (
            f"SymbolSets.{_symbol_set.name} contains invalid token 0x{_byte:02X} "
            f"(must be within symbol range 0x00..0xEF)"
        )


class SpecialToken(IntEnum):
    """Universal shared special tokens (0xF0..0xF7)."""

    DELIM = 0xF0  # | generic section delimiter
    PAD = 0xF1  # [PAD] padding token for batching
    BOS = 0xF2  # [BOS] beginning-of-sequence marker
    EOS = 0xF3  # [EOS] end-of-sequence marker
    QUERY_STEP = 0xF4  # [S] prompt tail marker where the step output is queried
    NOTIFY_CONTINUE = 0xF5  # [CONTINUE] loop continuation flag
    NOTIFY_FINISHED = 0xF6  # [FINISHED] loop termination flag


for _tok in SpecialToken:
    assert _tok in SHARED_SPECIAL_RANGE, (
        f"SpecialToken {_tok.name} not in {SHARED_SPECIAL_RANGE}"
    )


# ---------------------------------------------------------------------------
# 3. Experiment Vocabulary Container & Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentVocab:
    """The complete vocabulary configuration for an experiment or problem instance."""

    symbols: tuple[Token, ...]
    special_tokens: tuple[Token, ...]

    @property
    def alphabet_size(self) -> int:
        """Number of active symbols in the task alphabet (e.g. base B)."""
        return len(self.symbols)

    @property
    def vocab_size(self) -> int:
        """Fixed size of the neural network's token space (embedding and logit dimension).

        In yessir, every token is a single byte, so the model always predicts across
        all 256 byte classes regardless of how small the active task alphabet is.
        """
        return VOCAB_SIZE


def create_vocab(
    symbols: TokenSequence | SymbolSets,
    *,
    sample_size: int | None = None,
    rng: random.Random | None = None,
    experiment_specials: Sequence[Token] | type[IntEnum] = (),
) -> ExperimentVocab:
    """Create and validate an ExperimentVocab from a pool of symbols.

    Sampling and permutation behaviour:
    - sample_size=None, rng=None:
        Preserves the exact sequence of `symbols` as provided.
    - sample_size=None, rng=<Random>:
        Permutes (shuffles) the entire sequence of `symbols`.
    - sample_size=k, rng=<Random>:
        Samples `k` distinct symbols uniformly without replacement from `symbols`.
    - sample_size=k, rng=None:
        Raises ValueError. An explicit `rng` is strictly required for any
        random sampling or shuffling to ensure experiment reproducibility.

    Validation:
    - All tokens in `symbols` must be strictly within the symbol range (0x00..0xEF).
    - If `sample_size` is specified, it must satisfy 2 <= sample_size <= len(symbols).
    - All tokens in `experiment_specials` must be in the experiment special range (0xF8..0xFF).
    """
    if sample_size is not None and rng is None:
        raise ValueError(
            "An explicit rng (random.Random) must be provided when sample_size is specified."
        )

    # 1. Validate the symbol pool
    symbol_list = list(symbols)
    for tok in symbol_list:
        if tok not in SYMBOL_RANGE:
            raise ValueError(
                f"Symbol token 0x{tok:02X} is not within symbol range (0x00..0xEF)."
            )

    # 2. Select / sample / permute symbols
    if sample_size is not None:
        assert rng is not None
        if not (2 <= sample_size <= len(symbol_list)):
            raise ValueError(
                f"sample_size must satisfy 2 <= sample_size <= {len(symbol_list)}, got {sample_size}"
            )
        selected_symbols = tuple(rng.sample(symbol_list, sample_size))
    elif rng is not None:
        shuffled = symbol_list.copy()
        rng.shuffle(shuffled)
        selected_symbols = tuple(shuffled)
    else:
        selected_symbols = tuple(symbol_list)

    if len(set(selected_symbols)) != len(selected_symbols):
        raise ValueError("Selected symbols must not contain duplicate tokens.")

    # 3. Validate and unify special tokens
    exp_specials: list[int] = (
        [int(member) for member in experiment_specials]
        if isinstance(experiment_specials, type)
        and issubclass(experiment_specials, IntEnum)
        else [int(t) for t in experiment_specials]
    )
    for tok in exp_specials:
        if tok not in EXPERIMENT_SPECIAL_RANGE:
            raise ValueError(
                f"Experiment special token 0x{tok:02X} is not within experiment special range (0xF8..0xFF)."
            )

    all_specials = tuple(int(tok) for tok in SpecialToken) + tuple(exp_specials)
    if len(set(all_specials)) != len(all_specials):
        raise ValueError(
            "Special tokens contain duplicates between shared and experiment specials."
        )

    return ExperimentVocab(symbols=selected_symbols, special_tokens=all_specials)
