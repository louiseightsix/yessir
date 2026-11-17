"""Configuration schemas, evaluation-split specs, and cross-parameter validation for `addition`."""

import math
from dataclasses import dataclass, field
from enum import StrEnum, unique

from experiments.addition.inputs import EmbeddingMode, StepOrder
from yessir.alphabet import SymbolSet
from yessir.defaults import DEFAULT_LR, default_eval_every
from yessir.experiment import ExperimentConfig
from yessir.vocab import MAX_TOKEN_TYPES, VOCAB_SIZE

ADDITION_DESCRIPTION = (
    "In-context multi-base column addition and zero-shot base/length transfer."
)
CLS_DESCRIPTION = (
    "Per-step parallel classification variant read through ParallelClassificationHead."
)
STEP_FORK_DESCRIPTION = (
    "Per-step parallel branching queries ([S], [C], [F]) read through SubspaceLMHead."
)
STEP_AR_DESCRIPTION = (
    "Per-step sequential next-token prediction variant read through SubspaceLMHead."
)
AR_DESCRIPTION = "Full-trace autoregressive next-token prediction variant read through SubspaceLMHead."
ADDRESSED_STEP_AR_DESCRIPTION = "Explicitly base-B addressed per-step sequential next-token prediction variant read through SubspaceLMHead."
ADDRESSED_AR_DESCRIPTION = "Explicitly base-B addressed full-trace autoregressive next-token prediction variant read through SubspaceLMHead."


@unique
class SymbolSetName(StrEnum):
    """CLI/JSON-friendly names for the symbol pools in `yessir.alphabet.SymbolSet`."""

    ALL = "all"
    DECIMAL = "decimal"
    HEX = "hex"

    def resolve(self) -> SymbolSet:
        """Return the corresponding `SymbolSet` byte pool."""
        return SymbolSet[self.name]


@dataclass(frozen=True, kw_only=True)
class EvalSplit:
    """Distribution parameters for one evaluation split run on a trained model."""

    name: str
    prefix: str
    bases: tuple[int, ...]
    min_length: int
    max_length: int
    p_prior: float
    permuted_canonical: bool
    p_chain: float
    seed: int | str


def parse_bases_csv(raw: str, *, field_name: str) -> tuple[int, ...]:
    """Parse a comma-separated string like `'2,4,8,10'` into a validated tuple of distinct bases `>= 2`."""
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a str, got {type(raw).__name__}")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} must contain at least 1 base")

    values: list[int] = []
    for part in parts:
        try:
            base = int(part)
        except ValueError as exc:
            raise ValueError(
                f"invalid integer {part!r} in {field_name}={raw!r}"
            ) from exc
        if base < 2:
            raise ValueError(
                f"each base in {field_name} must be >= 2, got {base} in {raw!r}"
            )
        values.append(base)

    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates, got {raw!r}")
    return tuple(values)


@dataclass(frozen=True, kw_only=True)
class AdditionConfig(ExperimentConfig):
    """Validated inventory of settings shared across `addition` variants.

    Why multi-base fields (`bases`, `eval_ood_bases`) are stored as comma-separated
    `str` values rather than tuples/lists:
    - `ExperimentConfig` CLI/TUI introspection requires single-token scalar parsers,
      and `web/` diffs `config.json` values against variant defaults with `===`
      (which would treat JSON arrays `[2, 4, 8, 10]` as changed on every run).
    """

    # Task & training distribution
    bases: str = field(
        default="2,4,8,10", metadata={"domain": "comma-separated ints >= 2"}
    )
    min_length: int = field(default=1, metadata={"domain": "int >= 1"})
    max_length: int = field(default=20, metadata={"domain": "int >= min_length"})
    symbol_set: SymbolSetName = SymbolSetName.ALL
    symbol_pool_size: int | None = field(
        default=None, metadata={"domain": "int in 2..len(symbol_set) or None"}
    )
    p_prior: float = field(
        default=0.0,
        metadata={"domain": "float in 0.0..1.0", "step": 0.05, "min": 0.0, "max": 1.0},
    )
    p_chain: float = field(
        default=0.25,
        metadata={"domain": "float in 0.0..1.0", "step": 0.05, "min": 0.0, "max": 1.0},
    )

    # Model architecture (`ar` / `addressed_ar` pure-LM base; `AdditionStepConfig` adds harness embedding slices)
    n_layers: int = field(default=6, metadata={"domain": "int >= 1"})
    n_heads: int = field(default=8, metadata={"domain": "int >= 1 (divides d_model)"})
    d_ff: int = field(default=1376, metadata={"domain": "int >= 1", "step": 32})
    d_scratch: int = field(default=256, metadata={"domain": "int >= 0", "step": 16})

    # Training & logging
    max_steps: int = field(
        default=100_000, metadata={"domain": "int >= 1", "step": 1000}
    )
    batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    lr: float = field(
        default=DEFAULT_LR, metadata={"domain": "float > 0", "step": 1e-4}
    )
    log_every: int = field(default=100, metadata={"domain": "int >= 1", "step": 10})
    seed: int = field(default=42, metadata={"domain": "int (!= eval_seed)"})

    # Evaluation splits
    eval_every: int | None = field(
        default=None,
        metadata={
            "domain": "int >= 0 or None (None = max(2500, max_steps // 20); 0 = final only)",
            "step": 500,
            "min": 0,
        },
    )
    num_eval_problems: int = field(
        default=200, metadata={"domain": "int >= 1", "step": 50}
    )
    eval_batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    eval_seed: int = field(default=999, metadata={"domain": "int (!= seed)"})
    eval_p_chain: float = field(
        default=0.25,
        metadata={"domain": "float in 0.0..1.0", "step": 0.05, "min": 0.0, "max": 1.0},
    )
    eval_ood_bases: str | None = field(
        default="3,12", metadata={"domain": "comma-separated ints >= 2 or None"}
    )
    eval_extrap_max_length: int | None = field(
        default=40, metadata={"domain": "int > max_length or None", "step": 5}
    )
    eval_permuted_canonical: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol_set", SymbolSetName(self.symbol_set))

        train_bases = parse_bases_csv(self.bases, field_name="bases")
        object.__setattr__(self, "bases", ",".join(str(b) for b in train_bases))

        if self.eval_ood_bases is not None and not self.eval_ood_bases.strip():
            object.__setattr__(self, "eval_ood_bases", None)
        if self.eval_ood_bases is not None:
            # A base that is also trained on is not out of distribution, so it
            # gets no OOD split. Drop it from the stored value too, so that
            # `config.json` lists exactly the OOD bases that were evaluated.
            ood_bases = tuple(
                b
                for b in parse_bases_csv(
                    self.eval_ood_bases, field_name="eval_ood_bases"
                )
                if b not in train_bases
            )
            object.__setattr__(
                self,
                "eval_ood_bases",
                ",".join(str(b) for b in ood_bases) if ood_bases else None,
            )
        else:
            ood_bases = ()

        full_pool = self.symbol_set.resolve()
        if self.symbol_pool_size is not None and not (
            2 <= self.symbol_pool_size <= len(full_pool)
        ):
            raise ValueError(
                f"symbol_pool_size must be in 2..{len(full_pool)} for "
                f"symbol_set={self.symbol_set.value!r}, got {self.symbol_pool_size}"
            )

        pool_len = len(self.symbol_pool)
        max_train_base = max(train_bases)
        if self.p_prior < 1.0 and max_train_base > pool_len:
            raise ValueError(
                f"max(bases) ({max_train_base}) exceeds symbol_pool size ({pool_len})"
            )
        if self.p_prior > 0.0 and max_train_base > len(SymbolSet.HEX):
            raise ValueError(
                f"p_prior > 0 requires bases <= {len(SymbolSet.HEX)} (SymbolSet.HEX), "
                f"got {max_train_base}"
            )
        if self.eval_permuted_canonical and max_train_base > len(SymbolSet.HEX):
            raise ValueError(
                f"eval_permuted_canonical=True requires bases <= {len(SymbolSet.HEX)} "
                f"(SymbolSet.HEX), got {max_train_base}"
            )

        for b in ood_bases:
            if b not in train_bases and b > pool_len:
                raise ValueError(
                    f"eval_ood_bases entry {b} exceeds symbol_pool size ({pool_len})"
                )

        if self.min_length < 1:
            raise ValueError(f"min_length must be >= 1, got {self.min_length}")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be >= min_length ({self.min_length})"
            )
        if (
            self.eval_extrap_max_length is not None
            and self.eval_extrap_max_length <= self.max_length
        ):
            raise ValueError(
                f"eval_extrap_max_length ({self.eval_extrap_max_length}) must be > "
                f"max_length ({self.max_length}), or None to disable"
            )

        if not (0.0 <= self.p_prior <= 1.0):
            raise ValueError(f"p_prior must be in 0.0..1.0, got {self.p_prior}")
        if not (0.0 <= self.p_chain <= 1.0):
            raise ValueError(f"p_chain must be in 0.0..1.0, got {self.p_chain}")
        if not (0.0 <= self.eval_p_chain <= 1.0):
            raise ValueError(
                f"eval_p_chain must be in 0.0..1.0, got {self.eval_p_chain}"
            )

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
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr must be > 0 and finite, got {self.lr}")
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
    def parsed_bases(self) -> tuple[int, ...]:
        """Validated tuple of training bases."""
        return parse_bases_csv(self.bases, field_name="bases")

    @property
    def parsed_eval_ood_bases(self) -> tuple[int, ...]:
        """Validated tuple of OOD evaluation bases not already in `parsed_bases`."""
        if self.eval_ood_bases is None:
            return ()
        train_set = set(self.parsed_bases)
        return tuple(
            b
            for b in parse_bases_csv(self.eval_ood_bases, field_name="eval_ood_bases")
            if b not in train_set
        )

    @property
    def symbol_pool(self) -> bytes:
        """Active byte symbol pool to draw random alphabets from."""
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
        """Position-slice width passed to `SubspaceEmbedding` (`0` for pure-LM `ar` / `addressed_ar`)."""
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

    def _split_seed(self, split_name: str) -> str:
        """Derive a domain-separated split seed invariant across integer training `seed` sweeps."""
        return f"{self.eval_seed}:eval:{split_name}"

    def eval_splits(self) -> tuple[EvalSplit, ...]:
        """Build the ordered sequence of evaluation splits (`id` first, followed by enabled OOD splits)."""
        train_bases = self.parsed_bases
        splits: list[EvalSplit] = [
            EvalSplit(
                name="id",
                prefix="",
                bases=train_bases,
                min_length=self.min_length,
                max_length=self.max_length,
                p_prior=self.p_prior,
                permuted_canonical=False,
                p_chain=self.eval_p_chain,
                seed=self._split_seed("id"),
            )
        ]
        for b in self.parsed_eval_ood_bases:
            split_name = f"ood_base{b}"
            splits.append(
                EvalSplit(
                    name=split_name,
                    prefix=f"{split_name}_",
                    bases=(b,),
                    min_length=self.min_length,
                    max_length=self.max_length,
                    p_prior=0.0,
                    permuted_canonical=False,
                    p_chain=self.eval_p_chain,
                    seed=self._split_seed(split_name),
                )
            )
        if self.eval_extrap_max_length is not None:
            splits.append(
                EvalSplit(
                    name="extrap_len",
                    prefix="extrap_len_",
                    bases=train_bases,
                    min_length=self.max_length + 1,
                    max_length=self.eval_extrap_max_length,
                    p_prior=self.p_prior,
                    permuted_canonical=False,
                    p_chain=self.eval_p_chain,
                    seed=self._split_seed("extrap_len"),
                )
            )
        if self.eval_permuted_canonical:
            # Exclude B = 2 whenever any B >= 3 training base is available: in
            # binary, 1 / (B - 1) == 1.0 makes every in-alphabet value error
            # unconditionally count as `prior_intrusion` (`spec/task.md` §3).
            permuted_bases = tuple(b for b in train_bases if b >= 3) or train_bases
            splits.append(
                EvalSplit(
                    name="permuted",
                    prefix="permuted_",
                    bases=permuted_bases,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    p_prior=0.0,
                    permuted_canonical=True,
                    p_chain=self.eval_p_chain,
                    seed=self._split_seed("permuted"),
                )
            )
        return tuple(splits)

    @property
    def effective_eval_every(self) -> int | None:
        """Periodic evaluation interval in steps (`None` when `eval_every == 0` disables mid-run checkpoints)."""
        if self.eval_every is None:
            return default_eval_every(self.max_steps)
        return self.eval_every if self.eval_every > 0 else None

    def derived_summary(self) -> str:
        """Derived quantities displayed in the interactive TUI status bar."""
        split_names = ",".join(s.name for s in self.eval_splits())
        eval_str = (
            str(self.effective_eval_every)
            if self.effective_eval_every is not None
            else "final"
        )
        return (
            f"d_model={self.d_model}, symbol_pool={len(self.symbol_pool)}, "
            f"eval_every={eval_str}, splits=[{split_names}]"
        )


@dataclass(frozen=True, kw_only=True)
class AdditionStepConfig(AdditionConfig):
    """Shared base config for harnessed step-by-step `addition` variants.

    Adds harness-supplied `embedding_mode`, `d_position`, `fixed_d_model` (keeping
    `d_model = 256 + 16 + d_position + d_scratch = 512` across modes by default),
    and step-sampling controls (`p_terminal`, `all_steps`).
    """

    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE
    d_position: int = field(
        default=32,
        metadata={"domain": "even int >= 0 (>0 if token_type_pe)", "step": 2},
    )
    d_scratch: int = field(default=208, metadata={"domain": "int >= 0", "step": 16})
    fixed_d_model: bool = True
    p_terminal: float = field(
        default=0.2,
        metadata={"domain": "float in 0.0..1.0", "step": 0.05, "min": 0.0, "max": 1.0},
    )
    all_steps: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding_mode", EmbeddingMode(self.embedding_mode))
        if self.d_position < 0 or self.d_position % 2 != 0:
            raise ValueError(
                f"d_position must be a non-negative even int, got {self.d_position}"
            )
        if self.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE and self.d_position == 0:
            raise ValueError(
                "d_position must be > 0 when embedding_mode is 'token_type_pe'"
            )
        super().__post_init__()
        if not (0.0 <= self.p_terminal <= 1.0):
            raise ValueError(f"p_terminal must be in 0.0..1.0, got {self.p_terminal}")

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


@dataclass(frozen=True, kw_only=True)
class AdditionClsConfig(AdditionStepConfig):
    """Configuration schema for the `cls` variant of `addition`."""


@dataclass(frozen=True, kw_only=True)
class AdditionStepForkConfig(AdditionStepConfig):
    """Configuration schema for the `step_fork` variant of `addition`."""


@dataclass(frozen=True, kw_only=True)
class AdditionStepArConfig(AdditionStepConfig):
    """Configuration schema for the `step_ar` variant of `addition`."""

    step_order: StepOrder = StepOrder.SUM_FIRST
    use_query_tokens: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_order", StepOrder(self.step_order))
        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class AdditionArConfig(AdditionConfig):
    """Configuration schema for the pure-LM full-trace `ar` variant of `addition` (`input_ids` only)."""

    step_order: StepOrder = StepOrder.SUM_FIRST
    use_query_tokens: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_order", StepOrder(self.step_order))
        super().__post_init__()


def _validate_zero_preamble_lengths(
    config: AdditionConfig, *, include_alphabet: bool
) -> None:
    """Validate that `max_length >= B` across training and OOD bases when `include_alphabet=False`."""
    if include_alphabet:
        return
    max_train_base = max(config.parsed_bases)
    if config.max_length < max_train_base:
        raise ValueError(
            f"include_alphabet=False requires max_length ({config.max_length}) >= "
            f"max(bases) ({max_train_base}) so every alphabet symbol appears in operand addresses"
        )
    for b in config.parsed_eval_ood_bases:
        if config.max_length < b:
            raise ValueError(
                f"include_alphabet=False requires max_length ({config.max_length}) >= "
                f"eval_ood_bases entry ({b}) so OOD splits can sample L >= B"
            )


@dataclass(frozen=True, kw_only=True)
class AdditionAddressedStepArConfig(AdditionStepArConfig):
    """Configuration schema for the `addressed_step_ar` variant of `addition`.

    Defaults `embedding_mode` to `EmbeddingMode.NOPE` so explicit `[IDX] addr(j)`
    base-`B` addresses are not confounded with sinusoidal `position_ids` out of the
    box, while still allowing `--embedding-mode token_type_pe` / `token_type_nope`
    for step-harness ablations.
    """

    embedding_mode: EmbeddingMode = EmbeddingMode.NOPE
    include_alphabet: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_zero_preamble_lengths(self, include_alphabet=self.include_alphabet)


@dataclass(frozen=True, kw_only=True)
class AdditionAddressedArConfig(AdditionArConfig):
    """Configuration schema for the pure-LM full-trace `addressed_ar` variant of `addition` (`input_ids` only)."""

    include_alphabet: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_zero_preamble_lengths(self, include_alphabet=self.include_alphabet)
