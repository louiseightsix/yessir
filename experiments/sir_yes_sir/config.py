"""Configuration schema and cross-parameter validation for `sir_yes_sir` (`SirYesSirConfig`)."""

from dataclasses import dataclass, field
from enum import StrEnum, unique

from experiments.sir_yes_sir.inputs import EmbeddingMode
from yessir.alphabet import SymbolSet
from yessir.experiment import ExperimentConfig
from yessir.training import DEFAULT_LR
from yessir.vocab import MAX_TOKEN_TYPES, VOCAB_SIZE


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
class SirYesSirConfig(ExperimentConfig):
    """Flat, validated inventory of settings for a `sir_yes_sir` run.

    Cross-parameter rules enforced in `__post_init__`:
    1. **Symbol pool & alphabet size (`B`)**: `symbol_pool_size` (if given) slices the
       first `symbol_pool_size` bytes of `symbol_set` (useful for small-pool smoke
       tests on CPU), and `alphabet_size` must satisfy `2 <= alphabet_size <= len(symbol_pool)`.
    2. **Embedding mode vs. position dimensions**:
       - `EmbeddingMode.TOKEN_TYPE_PE` attaches `position_ids`, so `d_position` must
         be a positive even integer.
       - `EmbeddingMode.NOPE` and `EmbeddingMode.TOKEN_TYPE_NOPE` omit `position_ids`
         (`position_ids=None`), which `SubspaceEmbedding` requires to be paired with
         `d_position == 0`.
       - `use_token_type` is not a separate knob because it is completely determined by
         `embedding_mode` (`False` for `NOPE`, `True` for `TOKEN_TYPE_NOPE` and
         `TOKEN_TYPE_PE`).
    3. **Attention head divisibility**: `d_model` (`256 + d_token_type + d_position + d_scratch`)
       must be divisible by `n_heads`.
    4. **Disjoint train/eval streams**: `eval_seed` must differ from `seed` so evaluation
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
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE

    # Model architecture
    n_layers: int = field(default=6, metadata={"domain": "int >= 1"})
    n_heads: int = field(default=8, metadata={"domain": "int >= 1 (divides d_model)"})
    d_ff: int = field(default=1376, metadata={"domain": "int >= 1", "step": 32})
    d_position: int = field(
        default=32,
        metadata={"domain": "even int >= 0 (>0 if pe, 0 if nope)", "step": 2},
    )
    d_scratch: int = field(default=208, metadata={"domain": "int >= 0", "step": 16})

    # Training & evaluation
    steps: int = field(default=1000, metadata={"domain": "int >= 1", "step": 100})
    batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    lr: float = field(
        default=DEFAULT_LR, metadata={"domain": "float > 0", "step": 1e-4}
    )
    seed: int = field(default=42, metadata={"domain": "int (!= eval_seed)"})
    num_eval_problems: int = field(
        default=100, metadata={"domain": "int >= 1", "step": 10}
    )
    eval_batch_size: int = field(default=64, metadata={"domain": "int >= 1", "step": 8})
    eval_seed: int = field(default=999, metadata={"domain": "int (!= seed)"})

    def __post_init__(self) -> None:
        # Coerce string enum values when constructed from deserialised JSON/dicts.
        object.__setattr__(self, "symbol_set", SymbolSetName(self.symbol_set))
        object.__setattr__(self, "embedding_mode", EmbeddingMode(self.embedding_mode))

        full_pool = self.symbol_set.resolve()
        if self.symbol_pool_size is not None and not (
            2 <= self.symbol_pool_size <= len(full_pool)
        ):
            raise ValueError(
                f"symbol_pool_size must be in 2..{len(full_pool)} for "
                f"symbol_set={self.symbol_set.value!r}, got {self.symbol_pool_size}"
            )

        pool_len = len(self.symbol_pool)
        if not (2 <= self.alphabet_size <= pool_len):
            raise ValueError(
                f"alphabet_size must be in 2..{pool_len} (size of symbol_pool), "
                f"got {self.alphabet_size}"
            )
        if self.x_length < 1:
            raise ValueError(f"x_length must be >= 1, got {self.x_length}")

        if self.d_position < 0 or self.d_position % 2 != 0:
            raise ValueError(
                f"d_position must be a non-negative even int, got {self.d_position}"
            )
        if self.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE and self.d_position == 0:
            raise ValueError(
                "d_position must be > 0 when embedding_mode is 'token_type_pe'"
            )
        if self.embedding_mode != EmbeddingMode.TOKEN_TYPE_PE and self.d_position != 0:
            raise ValueError(
                f"d_position must be 0 when embedding_mode is {self.embedding_mode.value!r}, "
                f"got {self.d_position}"
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

        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
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
        return self.embedding_mode != EmbeddingMode.NOPE

    @property
    def d_model(self) -> int:
        """Total residual-stream width (`256 + d_token_type + d_position + d_scratch`)."""
        d_token_type = MAX_TOKEN_TYPES if self.use_token_type else 0
        return VOCAB_SIZE + d_token_type + self.d_position + self.d_scratch

    def derived_summary(self) -> str:
        """Derived quantities displayed in the interactive TUI status bar."""
        return f"d_model={self.d_model}, symbol_pool_size={len(self.symbol_pool)}"
