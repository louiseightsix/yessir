"""Pre-LN Transformer building blocks (`RMSNorm`, `SwiGLUFeedForward`, `MultiHeadSelfAttention`, `TransformerBlock`)."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation with an optional learned per-dimension gain.

    Computes `x / sqrt(mean(x^2) + eps)` (scaled by `weight` when `gain=True`).
    Unlike `LayerNorm`, `RMSNorm` does not subtract the mean, preserving the zero
    origin of the residual stream. The mean-square and reciprocal square root are
    computed in `float32` to avoid low-precision rounding in `bfloat16` (and
    overflow/underflow in `float16`).
    """

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-6,
        gain: bool = True,  # Equivalent to `elementwise_affine` in `torch.nn.RMSNorm`
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self._eps = eps
        self.weight: nn.Parameter | None
        if gain:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension of `x` of shape `(..., d_model)`."""
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(
            x_fp32.pow(2).mean(dim=-1, keepdim=True) + self._eps
        )
        if self.weight is not None:
            normed = normed * self.weight
        return normed.to(dtype=x.dtype)


class SwiGLUFeedForward(nn.Module):
    """Bias-free SwiGLU feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

    The elementwise product `silu(gate_proj(x)) * up_proj(x)` lets hidden units act as
    multiplicative gates, and omitting biases avoids adding constant offsets into
    the residual stream.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {d_ff}")

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN to `x` of shape `(B, T, d_model)`."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention using `F.scaled_dot_product_attention`.

    Why `is_causal=True` with no padding mask:
    In a right-padded `Batch`, real tokens sit at indices `0 .. length - 1` and
    `PAD` tokens sit at `>= length`. Under a causal mask, no position can attend
    to a later position, so real tokens never attend to `PAD` tokens and their
    representations are unaffected by padding.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self._n_heads = n_heads
        self._d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention to `x` of shape `(B, T, d_model)`."""
        batch_size, seq_len, d_model = x.shape
        head_shape = (batch_size, seq_len, self._n_heads, self._d_head)

        q = self.q_proj(x).view(head_shape).transpose(1, 2)
        k = self.k_proj(x).view(head_shape).transpose(1, 2)
        v = self.v_proj(x).view(head_shape).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        merged = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block (`x + attn(norm(x))` followed by `x + ffn(norm(x))`).

    Normalising on the branch input rather than on the residual sum keeps an
    unrescaled identity path from `SubspaceEmbedding` through to the output head.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one Pre-LN attention + FFN block on `x` of shape `(B, T, d_model)`."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
