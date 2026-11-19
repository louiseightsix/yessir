"""Configuration schemas and cross-parameter validation for `sir_yes_sir` (`SirYesSirConfig`, `SirYesSirArConfig`, `SirYesSirClsConfig`)."""

import math
from dataclasses import dataclass, field
from enum import StrEnum, unique

from experiments.sir_yes_sir.inputs import EmbeddingMode
from yessir.alphabet import SymbolSetName
from yessir.defaults import (
    DEFAULT_LR,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MIN_LR_RATIO,
    DEFAULT_WEIGHT_DECAY,
)
from yessir.experiment import ExperimentConfig
from yessir.vocab import MAX_TOKEN_TYPES, VOCAB_SIZE

SIR_YES_SIR_MIN_EVAL_EVERY = 50

SIR_YES_SIR_DESCRIPTION = "In-context cyclic successor lookup smoke-test experiment."
CLS_DESCRIPTION = (
    "Parallel classification variant read through ParallelClassificationHead."
)
AR_DESCRIPTION = "Autoregressive variant read through SubspaceLMHead."


@unique
class ClsSupervision(StrEnum):
    """How `cls` training examples are supervised per problem (`studies/sir_yes_sir/cls-training-passes.json`)."""

    SINGLE_PASS = (
        "single_pass"  # 1 per-position example supervising all L query positions
    )
    ALL_STEPS = (
        "all_steps"  # L separate single-step examples (k = 0 .. L - 1) per problem
    )
    RANDOM_STEP = "random_step"  # 1 randomly sampled single-step example k per problem


@dataclass(frozen=True, kw_only=True)
class SirYesSirConfig(ExperimentConfig):
    """Shared base and pure-LM (`ar`) configuration schema for `sir_yes_sir`.

    Autoregressive runs (`ar`) operate purely on `input_ids` (`use_token_type=False`,
    `effective_d_position=0`, `d_scratch=256` -> `d_model=512`), passing no side-channel
    `token_type_ids` or `position_ids` during generation. Step-harnessed runs
    (`SirYesSirClsConfig`) add `embedding_mode`, `d_position`, `fixed_d_model`, and
    `supervision`.

    Cross-parameter rules enforced in `__post_init__`:
    1. **Symbol pool & alphabet size (`B`)**: `symbol_pool_size` (if given) slices the
       first `symbol_pool_size` bytes of `symbol_set` (useful for small-pool smoke
       tests on CPU), and `alphabet_size` must satisfy `2 <= alphabet_size <= len(symbol_pool)`.
    2. **Attention head divisibility**: `d_model` must be divisible by `n_heads`.
    3. **Disjoint train/eval streams**: `eval_seed` must differ from `seed` so evaluation
       does not replay the training problem stream.
    """

    # Task & data
    alphabet_size: int = field(
        default=4, metadata={"domain": "int in 2..len(symbol_pool)"}
    )
    x_length: int = field(default=3, metadata={"domain": "int >= 1"})
    symbol_set: SymbolSetName = SymbolSetName.ALL
    symbol_pool_size: int | None = field(
        default=None, metadata={"domain": "int in 2..len(symbol_set) or None"}
    )
    resample_alphabet: bool = True

    # Model architecture
    n_layers: int = field(default=6, metadata={"domain": "int >= 1"})
    n_heads: int = field(default=8, metadata={"domain": "int >= 1 (divides d_model)"})
    d_ff: int = field(default=1376, metadata={"domain": "int >= 1", "step": 32})
    d_scratch: int = field(default=256, metadata={"domain": "int >= 0", "step": 16})

    # Training & evaluation
    max_steps: int = field(default=1000, metadata={"domain": "int >= 1", "step": 100})
    batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    grad_accum_steps: int = field(
        default=1, metadata={"domain": "int >= 1 (divides batch_size)", "min": 1}
    )
    lr: float = field(
        default=DEFAULT_LR, metadata={"domain": "float > 0", "step": 1e-4}
    )
    weight_decay: float = field(
        default=DEFAULT_WEIGHT_DECAY,
        metadata={"domain": "float >= 0.0", "step": 0.05, "min": 0.0},
    )
    warmup_steps: int = field(
        default=50, metadata={"domain": "int >= 0", "step": 10, "min": 0}
    )
    min_lr_ratio: float = field(
        default=DEFAULT_MIN_LR_RATIO,
        metadata={"domain": "float in 0.0..1.0", "step": 0.05, "min": 0.0, "max": 1.0},
    )
    max_grad_norm: float | None = field(
        default=DEFAULT_MAX_GRAD_NORM,
        metadata={"domain": "float > 0 or None", "step": 0.5, "min": 0.1},
    )
    log_every: int = field(default=10, metadata={"domain": "int >= 1", "step": 10})
    seed: int = field(default=42, metadata={"domain": "int (!= eval_seed)"})
    eval_every: int | None = field(
        default=None,
        metadata={
            "domain": "int >= 0 or None (None = max(50, max_steps // 20); 0 = final only)",
            "step": 50,
            "min": 0,
        },
    )
    num_eval_problems: int = field(
        default=100, metadata={"domain": "int >= 1", "step": 10}
    )
    eval_batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    eval_seed: int = field(default=999, metadata={"domain": "int (!= seed)"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol_set", SymbolSetName(self.symbol_set))

        full_pool = self.symbol_set.resolve()
        if self.symbol_pool_size is not None and not (
            2 <= self.symbol_pool_size <= len(full_pool)
        ):
            raise ValueError(
                f"symbol_pool_size must be in 2..{len(full_pool)} for "
                f"symbol_set={self.symbol_set.value!r}, got {self.symbol_pool_size}"
            )
        if not self.resample_alphabet and self.symbol_pool_size is not None:
            raise ValueError(
                f"symbol_pool_size is unused when resample_alphabet=False "
                f"(uses symbol_set[:alphabet_size]); leave symbol_pool_size=None, got {self.symbol_pool_size}"
            )

        pool_len = len(self.symbol_pool)
        if not (2 <= self.alphabet_size <= pool_len):
            raise ValueError(
                f"alphabet_size must be in 2..{pool_len} (size of symbol_pool), "
                f"got {self.alphabet_size}"
            )
        if self.x_length < 1:
            raise ValueError(f"x_length must be >= 1, got {self.x_length}")

        if self.d_scratch < 0:
            raise ValueError(f"d_scratch must be >= 0, got {self.d_scratch}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {self.d_ff}")

        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if self.batch_size % self.grad_accum_steps != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"grad_accum_steps ({self.grad_accum_steps})"
            )
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr must be > 0 and finite, got {self.lr}")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0 and finite, got {self.weight_decay}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not math.isfinite(self.min_lr_ratio) or not (
            0.0 <= self.min_lr_ratio <= 1.0
        ):
            raise ValueError(
                f"min_lr_ratio must be in 0.0..1.0 and finite, got {self.min_lr_ratio}"
            )
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError(
                f"max_grad_norm must be > 0 and finite (or None), got {self.max_grad_norm}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.eval_every is not None and (
            isinstance(self.eval_every, bool)
            or not isinstance(self.eval_every, int)
            or self.eval_every < 0
        ):
            raise ValueError(
                f"eval_every must be an int >= 0 or None, got {self.eval_every!r}"
            )
        if self.num_eval_problems < 1:
            raise ValueError(
                f"num_eval_problems must be >= 1, got {self.num_eval_problems}"
            )
        if self.eval_batch_size < 1:
            raise ValueError(
                f"eval_batch_size must be >= 1, got {self.eval_batch_size}"
            )
        if self.eval_seed == self.seed:
            raise ValueError(
                f"eval_seed ({self.eval_seed}) must differ from training seed ({self.seed})"
            )

    @property
    def symbol_pool(self) -> bytes:
        """The active byte symbol pool to draw alphabets from."""
        pool = bytes(self.symbol_set.resolve())
        if self.symbol_pool_size is not None:
            return pool[: self.symbol_pool_size]
        return pool

    @property
    def use_token_type(self) -> bool:
        """Whether `SubspaceEmbedding` includes the 16-d `token_type_ids` slice."""
        return False

    @property
    def effective_d_position(self) -> int:
        """Position-slice width passed to `SubspaceEmbedding` (`0` for pure-LM `ar`)."""
        return 0

    @property
    def effective_d_scratch(self) -> int:
        """Scratch-slice width passed to `SubspaceEmbedding`."""
        return self.d_scratch

    @property
    def embedding_mode_label(self) -> str:
        """Human-readable embedding mode name for run startup logs."""
        return EmbeddingMode.NOPE.value

    @property
    def d_model(self) -> int:
        """Total residual-stream width (`256 + d_token_type + effective_d_position + effective_d_scratch`)."""
        d_token_type = MAX_TOKEN_TYPES if self.use_token_type else 0
        return (
            VOCAB_SIZE
            + d_token_type
            + self.effective_d_position
            + self.effective_d_scratch
        )

    @property
    def effective_eval_every(self) -> int | None:
        """Periodic evaluation interval in steps (`None` when `eval_every == 0` disables mid-run checkpoints)."""
        if self.eval_every is None:
            return max(SIR_YES_SIR_MIN_EVAL_EVERY, self.max_steps // 20)
        return self.eval_every if self.eval_every > 0 else None

    def derived_summary(self) -> str:
        """Derived quantities displayed in the interactive TUI status bar."""
        eval_str = (
            str(self.effective_eval_every)
            if self.effective_eval_every is not None
            else "final"
        )
        return (
            f"d_model={self.d_model} (d_pos={self.effective_d_position}, "
            f"d_scratch={self.effective_d_scratch}), symbol_pool_size={len(self.symbol_pool)}, "
            f"eval_every={eval_str}"
        )


@dataclass(frozen=True, kw_only=True)
class SirYesSirArConfig(SirYesSirConfig):
    """Configuration schema for the pure-LM `ar` variant of `sir_yes_sir` (`input_ids` only)."""


@dataclass(frozen=True, kw_only=True)
class SirYesSirClsConfig(SirYesSirConfig):
    """Configuration schema for the harnessed `cls` variant of `sir_yes_sir`.

    Adds harness-supplied `embedding_mode`, `d_position`, `fixed_d_model` (keeping
    `d_model = 256 + 16 + d_position + d_scratch = 512` across modes by default),
    and `supervision`.
    """

    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE
    d_position: int = field(
        default=32,
        metadata={"domain": "even int >= 4 (32 unless token_type_pe)", "step": 2},
    )
    d_scratch: int = field(default=208, metadata={"domain": "int >= 0", "step": 16})
    fixed_d_model: bool = True
    supervision: ClsSupervision = ClsSupervision.SINGLE_PASS

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding_mode", EmbeddingMode(self.embedding_mode))
        object.__setattr__(self, "supervision", ClsSupervision(self.supervision))
        if self.d_position < 0 or self.d_position % 2 != 0:
            raise ValueError(
                f"d_position must be a non-negative even int, got {self.d_position}"
            )
        if self.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE:
            if self.d_position < 4:
                raise ValueError(
                    f"d_position must be >= 4 when embedding_mode is 'token_type_pe' "
                    f"(d_position=2 has single period 2*pi ~ 6.28), got {self.d_position}"
                )
        elif self.d_position != 32:
            raise ValueError(
                f"d_position is unused when embedding_mode={self.embedding_mode.value!r}; "
                f"leave d_position=32 (got {self.d_position}) and adjust d_scratch or fixed_d_model instead"
            )
        super().__post_init__()

    @property
    def use_token_type(self) -> bool:
        return self.embedding_mode != EmbeddingMode.NOPE

    @property
    def effective_d_position(self) -> int:
        return (
            self.d_position if self.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE else 0
        )

    @property
    def effective_d_scratch(self) -> int:
        if not self.fixed_d_model:
            return self.d_scratch
        dropped_token_type = 0 if self.use_token_type else MAX_TOKEN_TYPES
        dropped_position = (
            0 if self.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE else self.d_position
        )
        return self.d_scratch + dropped_token_type + dropped_position

    @property
    def embedding_mode_label(self) -> str:
        return self.embedding_mode.value
