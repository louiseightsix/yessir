"""Byte-level vocabulary definitions, token sets, and alphabet sampling."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

# ---------------------------------------------------------------------------
# 1. Byte Vocabulary Partition Geometry
# ---------------------------------------------------------------------------
#
# The 256-byte vocabulary is partitioned into three contiguous segments:
#
#   0x00                     0xEF  0xF0            0xF7  0xF8            0xFF
#  ┌─────────────────────────────┬─────────────────────┬─────────────────────┐
#  │       Literal Tokens        │   Shared Control    │ Experiment Control  │
#  │         (240 bytes)         │      (8 bytes)      │      (8 bytes)      │
#  └─────────────────────────────┴─────────────────────┴─────────────────────┘
#  0                             _SHARED_START         _EXPERIMENT_START     256
#                                (240 / 0xF0)          (248 / 0xF8)
#
_YESSIR_VOCAB_SIZE = 256

_NUM_SHARED_CONTROL = 8
_NUM_EXPERIMENT_CONTROL = 8
_NUM_LITERALS = _YESSIR_VOCAB_SIZE - _NUM_SHARED_CONTROL - _NUM_EXPERIMENT_CONTROL

_SHARED_START = _NUM_LITERALS
_EXPERIMENT_START = _NUM_LITERALS + _NUM_SHARED_CONTROL

_LITERAL_RANGE = range(_SHARED_START)
_SHARED_RANGE = range(_SHARED_START, _EXPERIMENT_START)
_EXPERIMENT_RANGE = range(_EXPERIMENT_START, _YESSIR_VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 2. Token Enums
# ---------------------------------------------------------------------------


class LiteralTokenSets(bytes, Enum):
    """Predefined collections of literal tokens for alphabet selection."""

    ALL = bytes(_LITERAL_RANGE)
    DECIMAL = b"0123456789"
    HEX = b"0123456789ABCDEF"


# Static validation: ensure every predefined literal set contains only valid literal bytes
for _token_set in LiteralTokenSets:
    for _byte in _token_set.value:
        assert _byte in _LITERAL_RANGE, (
            f"LiteralTokenSets.{_token_set.name} contains invalid token 0x{_byte:02X} "
            f"(must be within literal range 0x00..0xEF)"
        )


class SharedControlToken(IntEnum):
    """Universal shared control tokens (0xF0..0xF7)."""

    DELIM = 0xF0  # | generic section delimiter
    PAD = 0xF1  # [PAD] right-padding token for batched sequences
    BOS = 0xF2  # [BOS] beginning-of-sequence marker
    EOS = 0xF3  # [EOS] end-of-sequence marker


for _tok in SharedControlToken:
    assert _tok in _SHARED_RANGE, (
        f"SharedControlToken {_tok.name} not in {_SHARED_RANGE}"
    )


# ---------------------------------------------------------------------------
# 3. Experiment Vocabulary Container & Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentVocab:
    """The complete vocabulary configuration for an experiment or problem instance."""

    literals: tuple[int, ...]
    control_tokens: tuple[int, ...]

    @property
    def alphabet_size(self) -> int:
        """Number of active literal symbols in the task alphabet (e.g. base B)."""
        return len(self.literals)

    @property
    def model_vocab_size(self) -> int:
        """Fixed size of the neural network's token space (embedding and logit dimension).

        In yessir, every token is a single byte, so the model always predicts across
        all 256 byte classes regardless of how small the active task alphabet is.
        """
        return _YESSIR_VOCAB_SIZE


def create_vocab(
    literals: Sequence[int] | bytes | LiteralTokenSets,
    *,
    sample_size: int | None = None,
    rng: random.Random | None = None,
    experiment_controls: Sequence[int] | type[IntEnum] = (),
) -> ExperimentVocab:
    """Create and validate an ExperimentVocab from a pool of literal tokens.

    Sampling and permutation behaviour:
    - sample_size=None, rng=None:
        Preserves the exact sequence of `literals` as provided.
    - sample_size=None, rng=<Random>:
        Permutes (shuffles) the entire sequence of `literals`.
    - sample_size=k, rng=<Random>:
        Samples `k` distinct tokens uniformly without replacement from `literals`.
    - sample_size=k, rng=None:
        Raises ValueError. An explicit `rng` is strictly required for any
        random sampling or shuffling to ensure experiment reproducibility.

    Validation:
    - All tokens in `literals` must be strictly within the literal range (0x00..0xEF).
    - If `sample_size` is specified, it must satisfy 2 <= sample_size <= len(literals).
    - All tokens in `experiment_controls` must be in the experiment control range (0xF8..0xFF).
    """
    if sample_size is not None and rng is None:
        raise ValueError(
            "An explicit rng (random.Random) must be provided when sample_size is specified."
        )

    # 1. Validate the literal pool
    literal_list = list(literals)
    for tok in literal_list:
        if tok not in _LITERAL_RANGE:
            raise ValueError(
                f"Literal token 0x{tok:02X} is not within literal range (0x00..0xEF)."
            )

    # 2. Select / sample / permute literals
    if sample_size is not None:
        assert rng is not None
        if not (2 <= sample_size <= len(literal_list)):
            raise ValueError(
                f"sample_size must satisfy 2 <= sample_size <= {len(literal_list)}, got {sample_size}"
            )
        selected_literals = tuple(rng.sample(literal_list, sample_size))
    elif rng is not None:
        shuffled = literal_list.copy()
        rng.shuffle(shuffled)
        selected_literals = tuple(shuffled)
    else:
        selected_literals = tuple(literal_list)

    if len(set(selected_literals)) != len(selected_literals):
        raise ValueError("Selected literals must not contain duplicate tokens.")

    # 3. Validate and unify control tokens
    exp_ctrls: list[int] = (
        [int(member) for member in experiment_controls]
        if isinstance(experiment_controls, type)
        and issubclass(experiment_controls, IntEnum)
        else [int(t) for t in experiment_controls]
    )
    for tok in exp_ctrls:
        if tok not in _EXPERIMENT_RANGE:
            raise ValueError(
                f"Experiment control token 0x{tok:02X} is not within experiment control range (0xF8..0xFF)."
            )

    all_controls = tuple(int(tok) for tok in SharedControlToken) + tuple(exp_ctrls)
    if len(set(all_controls)) != len(all_controls):
        raise ValueError(
            "Control tokens contain duplicates between shared and experiment controls."
        )

    return ExperimentVocab(literals=selected_literals, control_tokens=all_controls)
