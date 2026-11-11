"""Parameter-free orthogonal subspace embedding (`SubspaceEmbedding`).

Maps each token position into a `d_model`-dimensional initial residual-stream
vector by concatenating up to four disjoint slices:

  0 .. 255                +16 (optional)         +d_position (optional)  +d_scratch (optional)
 ┌───────────────────────┬──────────────────────┬───────────────────────┬──────────────────────┐
 │ Symbol (`input_ids`)  │ Section (`type_ids`) │ Position (`pos_ids`)  │ Scratchpad (`0.0`s)  │
 │ 256-d frozen one-hot  │ 16-d frozen one-hot  │ sinusoidal (norm = 1) │ working memory       │
 └───────────────────────┴──────────────────────┴───────────────────────┴──────────────────────┘

Why parameter-free orthogonal slices:
1. All inputs (`256` byte tokens, `16` token types, and sinusoidal positions)
   are small enough to represent directly without a learned compression table,
   and keeping byte symbols as a frozen one-hot basis prevents symbols from
   acquiring fixed meanings across problems.
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

import torch
import torch.nn.functional as F
from torch import nn

from yessir.vocab import SPECIAL_RANGE, VOCAB_SIZE

_NUM_TOKEN_TYPES = len(SPECIAL_RANGE)  # 16
_SINUSOIDAL_BASE = 10_000.0


class SubspaceEmbedding(nn.Module):
    """Concatenates frozen one-hot symbol, optional one-hot token-type, optional sinusoidal position, and zero scratchpad slices."""

    _inv_freq: torch.Tensor

    def __init__(
        self,
        *,
        use_token_type: bool = True,
        d_position: int = 32,
        d_scratch: int = 208,
        dtype: torch.dtype = torch.float32,
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
        self._dtype = dtype

        d_token_type = _NUM_TOKEN_TYPES if use_token_type else 0
        self.d_model = VOCAB_SIZE + d_token_type + d_position + d_scratch

        if d_position > 0:
            # Standard Transformer geometric frequencies: omega_i = 10000^(-2i / d_position)
            # for i in 0 .. (d_position / 2) - 1.
            half_pos = d_position // 2
            inv_freq = 1.0 / (
                _SINUSOIDAL_BASE
                ** (torch.arange(0, half_pos, dtype=torch.float32) / half_pos)
            )
            self.register_buffer("_inv_freq", inv_freq, persistent=False)
            # Each (sin, cos) pair has sin^2 + cos^2 = 1, so `half_pos` pairs have
            # L2 norm sqrt(half_pos). Scaling by 1 / sqrt(half_pos) gives exact unit norm.
            self._position_scale = 1.0 / math.sqrt(half_pos)
        else:
            self.register_buffer("_inv_freq", torch.empty(0), persistent=False)
            self._position_scale = 0.0

    def _sinusoidal_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute unit-norm sinusoidal encodings of shape `(B, T, d_position)`."""
        angles = position_ids.unsqueeze(-1).float() * self._inv_freq  # (B, T, half_pos)
        sin_cos = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(
            -2
        )
        return (sin_cos * self._position_scale).to(dtype=self._dtype)

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

        slices: list[torch.Tensor] = [
            F.one_hot(input_ids, num_classes=VOCAB_SIZE).to(dtype=self._dtype)
        ]

        if self._use_token_type:
            assert token_type_ids is not None
            slices.append(
                F.one_hot(token_type_ids, num_classes=_NUM_TOKEN_TYPES).to(
                    dtype=self._dtype
                )
            )

        if self._d_position > 0:
            assert position_ids is not None
            slices.append(self._sinusoidal_positions(position_ids))

        if self._d_scratch > 0:
            slices.append(
                torch.zeros(
                    (*input_ids.shape, self._d_scratch),
                    device=input_ids.device,
                    dtype=self._dtype,
                )
            )

        return torch.cat(slices, dim=-1)
