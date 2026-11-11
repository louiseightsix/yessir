"""Parameter-free orthogonal subspace embedding (`SubspaceEmbedding`).

Maps each token position into a `d_model`-dimensional initial residual-stream
vector by concatenating up to four disjoint slices:

  0 .. 255                +16 (optional)            +d_position (optional)     +d_scratch (optional)
 ┌───────────────────────┬─────────────────────────┬──────────────────────────┬──────────────────────┐
 │ Symbol (`input_ids`)  │ Section                 │ Position                 │ Scratchpad (`0.0`s)  │
 │ 256-d frozen one-hot  │ (`token_type_ids`)      │ (`position_ids`)         │ working memory       │
 │ (norm = 1)            │ 16-d one-hot (norm = 1) │ sinusoidal (norm = 1)    │                      │
 └───────────────────────┴─────────────────────────┴──────────────────────────┴──────────────────────┘

Why parameter-free orthogonal slices:
1. All inputs (`256` byte tokens, up to `16` per-experiment section types, and
   sinusoidal positions) are small enough to represent directly without a
   learned compression table. The frozen one-hot symbol slice gives every byte
   equal initial norm (`1.0`), mutually orthogonal directions that never drift,
   direct interpretability across layers, and a fixed basis that a tied output
   head can read back without learned unembedding weights. (Note: a frozen
   one-hot does not by itself stop Layer 0's weights or `RMSNorm` gain from
   learning per-symbol biases—preventing fixed symbol meanings comes from
   sampling fresh alphabets across problems in the training data.)
2. Each active input slice (`symbol`, `token_type`, `position`) has exact L2
   norm `1.0`, so none dominates the first `RMSNorm`.
3. `d_scratch` reserves dedicated zero-initialised working-memory dimensions so
   intermediate layers do not have to overwrite input coordinates.
4. Omitted slices (`use_token_type=False` or `d_position=0`) are not included,
   so `d_scratch` always equals the exact number of zero-initialised scratchpad
   dimensions. To hold total `d_model` constant across modes, the caller adjusts
   `d_scratch` explicitly.
"""

import math
from typing import Annotated

import torch
import torch.nn.functional as F
from torch import nn

from yessir.vocab import VOCAB_SIZE

# Each experiment defines its own TokenType IntEnum (e.g. SPECIAL, ALPHABET,
# INPUT, OUT) with values in TOKEN_TYPE_RANGE (0 .. MAX_TOKEN_TYPES - 1).
MAX_TOKEN_TYPES = 16
TOKEN_TYPE_RANGE = range(0, MAX_TOKEN_TYPES)  # noqa: PIE808
type TokenTypeId = Annotated[int, "Value in 0..15 (TOKEN_TYPE_RANGE)"]

# In our experiments `position_ids` restart at 0 in every section (alphabet of
# size B <= 240, operands of length L <= 50, or an autoregressive trace of a few
# hundred tokens). Vaswani et al.'s base of 10,000 was chosen for sequences of
# thousands of tokens; with d_position = 32 (16 frequency pairs) and positions
# in 0..31, base 10,000 leaves the slower 9 pairs nearly frozen at (sin≈0, cos≈1).
# Base 256.0 (slowest half-wavelength pi * 256 ≈ 804 positions, full cycle ≈ 1,608)
# spreads all 16 frequency pairs across the section lengths we actually use.
_SINUSOIDAL_BASE = 256.0


class SubspaceEmbedding(nn.Module):
    """Concatenates frozen one-hot symbol, optional one-hot token-type, optional sinusoidal position, and zero scratchpad slices."""

    _dtype_tracker: torch.Tensor

    def __init__(
        self,
        *,
        use_token_type: bool = True,
        d_position: int = 32,
        d_scratch: int = 208,
    ) -> None:
        super().__init__()
        if d_position < 0 or d_position % 2 != 0:
            raise ValueError(
                f"d_position must be a non-negative even int, got {d_position}"
            )
        if d_scratch < 0:
            raise ValueError(f"d_scratch must be >= 0, got {d_scratch}")

        self._use_token_type = use_token_type
        self._d_position = d_position
        self._d_scratch = d_scratch

        d_token_type = MAX_TOKEN_TYPES if use_token_type else 0
        self.d_model = VOCAB_SIZE + d_token_type + d_position + d_scratch

        # Tracks module-level `.to(dtype=...)` calls without subjecting sinusoidal
        # frequencies to low-precision (e.g. bfloat16) rounding.
        self.register_buffer(
            "_dtype_tracker", torch.empty(0, dtype=torch.float32), persistent=False
        )

    def _sinusoidal_positions(
        self, position_ids: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute unit-norm sinusoidal encodings of shape `(B, T, d_position)` in float32."""
        half_pos = self._d_position // 2
        inv_freq = 1.0 / (
            _SINUSOIDAL_BASE
            ** (
                torch.arange(
                    0, half_pos, device=position_ids.device, dtype=torch.float32
                )
                / half_pos
            )
        )
        angles = position_ids.unsqueeze(-1).float() * inv_freq  # (B, T, half_pos)
        sin_cos = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(
            -2
        )
        # Each (sin, cos) pair has sin^2 + cos^2 = 1, so `half_pos` pairs have
        # L2 norm sqrt(half_pos). Scaling by 1 / sqrt(half_pos) gives exact unit norm.
        scale = 1.0 / math.sqrt(half_pos)
        return (sin_cos * scale).to(dtype=out_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed `(B, T)` integer tensors into `(B, T, d_model)` float representations."""
        if self._use_token_type and token_type_ids is None:
            raise ValueError("token_type_ids is required when use_token_type=True")
        if not self._use_token_type and token_type_ids is not None:
            raise ValueError("token_type_ids must be None when use_token_type=False")

        if self._d_position > 0 and position_ids is None:
            raise ValueError("position_ids is required when d_position > 0")
        if self._d_position == 0 and position_ids is not None:
            raise ValueError("position_ids must be None when d_position == 0")

        out_dtype = self._dtype_tracker.dtype

        slices: list[torch.Tensor] = [
            F.one_hot(input_ids, num_classes=VOCAB_SIZE).to(dtype=out_dtype)
        ]

        if self._use_token_type:
            assert token_type_ids is not None
            if ((token_type_ids < 0) | (token_type_ids >= MAX_TOKEN_TYPES)).any():
                raise ValueError(f"token_type_ids must be in 0..{MAX_TOKEN_TYPES - 1}")
            slices.append(
                F.one_hot(token_type_ids, num_classes=MAX_TOKEN_TYPES).to(
                    dtype=out_dtype
                )
            )

        if self._d_position > 0:
            assert position_ids is not None
            slices.append(self._sinusoidal_positions(position_ids, out_dtype))

        if self._d_scratch > 0:
            slices.append(
                torch.zeros(
                    (*input_ids.shape, self._d_scratch),
                    device=input_ids.device,
                    dtype=out_dtype,
                )
            )

        return torch.cat(slices, dim=-1)
