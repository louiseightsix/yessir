"""Byte-level vocabulary constants and alphabet sampling.

Every token in yessir is a single byte in 0x00..0xFF (VOCAB_SIZE = 256),
split into three non-overlapping ranges:

1. LITERAL_TOKENS (0x00..0xEF, 240 bytes):
   The reservoir of literal symbols from which task alphabets are sampled.
2. SHARED_CONTROL_TOKENS (0xF0..0xF7, 8 bytes):
   Universal framework control tokens shared across all experiments.
3. EXPERIMENT_CONTROL_TOKENS (0xF8..0xFF, 8 bytes):
   Reserved for experiment-specific control tokens (e.g. query triggers, task flags).
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

VOCAB_SIZE = 256

# The three non-overlapping token partitions of the 256-byte vocabulary:
LITERAL_TOKENS: range = range(0xF0)  # 0x00..0xEF (240 bytes)
SHARED_CONTROL_TOKENS: range = range(0xF0, 0xF8)  # 0xF0..0xF7 (8 bytes)
EXPERIMENT_CONTROL_TOKENS: range = range(0xF8, 0x100)  # 0xF8..0xFF (8 bytes)


class SharedControl(IntEnum):
    """Universal shared control tokens in SHARED_CONTROL_TOKENS (0xF0..0xF7)."""

    DELIM = 0xF0  # | generic section delimiter
    PAD = 0xF1  # [PAD] right-padding token for batched sequences
    BOS = 0xF2  # [BOS] beginning-of-sequence marker
    EOS = 0xF3  # [EOS] end-of-sequence marker


# Standard character sets as raw bytes
DECIMAL_TOKENS: bytes = b"0123456789"
HEX_TOKENS: bytes = b"0123456789ABCDEF"


@dataclass(frozen=True)
class ExperimentVocab:
    """The complete vocabulary configuration for an experiment or problem instance."""

    literals: tuple[int, ...]
    control_tokens: tuple[int, ...]

    @property
    def all_tokens(self) -> tuple[int, ...]:
        """Union of literals and all active control tokens."""
        return self.literals + self.control_tokens


def create_vocab(
    literals: Sequence[int] | bytes,
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
    - All tokens in `literals` must be strictly within LITERAL_TOKENS (0x00..0xEF).
    - If `sample_size` is specified, it must satisfy 2 <= sample_size <= len(literals).
    - All tokens in `experiment_controls` must be in EXPERIMENT_CONTROL_TOKENS (0xF8..0xFF).
    """
    if sample_size is not None and rng is None:
        raise ValueError(
            "An explicit rng (random.Random) must be provided when sample_size is specified."
        )

    # 1. Validate the literal pool
    literal_list = list(literals)
    for tok in literal_list:
        if tok not in LITERAL_TOKENS:
            raise ValueError(
                f"Literal token 0x{tok:02X} is not within LITERAL_TOKENS (0x00..0xEF)."
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
        if tok not in EXPERIMENT_CONTROL_TOKENS:
            raise ValueError(
                f"Experiment control token 0x{tok:02X} is not within EXPERIMENT_CONTROL_TOKENS (0xF8..0xFF)."
            )

    all_controls = tuple(int(tok) for tok in SharedControl) + tuple(exp_ctrls)
    if len(set(all_controls)) != len(all_controls):
        raise ValueError(
            "Control tokens contain duplicates between shared and experiment controls."
        )

    return ExperimentVocab(literals=selected_literals, control_tokens=all_controls)
