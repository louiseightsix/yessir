"""Pre-LN Transformer building blocks (`RMSNorm`, `SwiGLUFeedForward`, `MultiHeadSelfAttention`, `TransformerBlock`) and inference KV cache (`KVCache`, `build_causal_attention_mask`)."""

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


def build_causal_attention_mask(
    key_padding_mask: torch.Tensor,
    *,
    start_pos: int = 0,
) -> torch.Tensor:
    """Build a `(B, 1, q_len, k_len)` boolean causal mask from `(B, k_len)` key validity.

    Real queries attend to valid keys (`key_padding_mask == True`) at positions
    `<= q_pos`. Left-padded queries (`key_padding_mask == False`) attend to their
    own diagonal slot (`k_pos == q_pos`) so softmax stays finite across all layers
    without leaking padded keys into any real query.
    """
    if key_padding_mask.ndim != 2 or key_padding_mask.dtype != torch.bool:
        raise ValueError(
            "key_padding_mask must be a 2D bool tensor of shape (batch_size, k_len)"
        )
    k_len = int(key_padding_mask.size(1))
    if not (0 <= start_pos <= k_len):
        raise ValueError(f"start_pos ({start_pos}) must lie in 0..{k_len}")

    device = key_padding_mask.device
    q_pos = torch.arange(start_pos, k_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    causal = k_pos <= q_pos
    diag = k_pos == q_pos
    return (
        (causal.unsqueeze(0) & key_padding_mask[:, None, :]) | diag.unsqueeze(0)
    ).unsqueeze(1)


class KVCache:
    """Pre-allocated per-layer key/value cache for `@torch.no_grad()` autoregressive decoding.

    Stores `(B, n_heads, max_seq_len, d_head)` key and value buffers per layer
    (allocated lazily on the first forward pass to match runtime batch size, device,
    and dtype) alongside an optional `(B, max_seq_len)` boolean `key_padding_mask`
    for left-padded batched prompts.
    """

    def __init__(self, n_layers: int, max_seq_len: int) -> None:
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        self._n_layers = n_layers
        self.max_seq_len = max_seq_len
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers
        self._key_mask: torch.Tensor | None = None
        self._batch_size: int | None = None
        self._seq_len = 0
        self._start_pos = 0
        self._end_pos = 0

    @property
    def seq_len(self) -> int:
        """Number of token positions committed to the cache so far."""
        return self._seq_len

    @property
    def start_pos(self) -> int:
        """Starting sequence offset of the active forward pass."""
        return self._start_pos

    def prepare_step(
        self,
        batch_size: int,
        step_len: int,
        *,
        device: torch.device,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Prepare slice offsets and return the shared `(B, 1, step_len, end_pos)` mask (or `None`)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if step_len <= 0:
            raise ValueError(f"step_len must be > 0, got {step_len}")
        if self._batch_size is not None and batch_size != self._batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) does not match cached batch_size ({self._batch_size})"
            )

        start_pos = self._seq_len
        end_pos = start_pos + step_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"KVCache overflow: step [{start_pos}:{end_pos}] exceeds max_seq_len={self.max_seq_len}"
            )

        if key_padding_mask is not None and (
            key_padding_mask.shape != (batch_size, step_len)
            or key_padding_mask.dtype != torch.bool
        ):
            raise ValueError(
                f"key_padding_mask must be a bool tensor of shape {(batch_size, step_len)}"
            )

        if key_padding_mask is not None and self._key_mask is None:
            self._key_mask = torch.ones(
                (batch_size, self.max_seq_len), dtype=torch.bool, device=device
            )

        if self._key_mask is not None:
            if key_padding_mask is not None:
                self._key_mask[:, start_pos:end_pos] = key_padding_mask
            else:
                self._key_mask[:, start_pos:end_pos] = True

        self._batch_size = batch_size
        self._start_pos = start_pos
        self._end_pos = end_pos

        if self._key_mask is not None:
            return build_causal_attention_mask(
                self._key_mask[:, :end_pos], start_pos=start_pos
            )
        if start_pos > 0 and step_len > 1:
            q_pos = torch.arange(start_pos, end_pos, device=device)[:, None]
            k_pos = torch.arange(end_pos, device=device)[None, :]
            return (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
        return None

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write `(k, v)` for `layer_idx` into `[start_pos:end_pos]` and return `[:end_pos]`."""
        if not (0 <= layer_idx < self._n_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range 0..{self._n_layers - 1}"
            )
        if self._end_pos <= self._start_pos:
            raise RuntimeError("KVCache.prepare_step must be called before update")

        k_buf = self._keys[layer_idx]
        v_buf = self._values[layer_idx]
        if k_buf is None or v_buf is None:
            batch_size, n_heads, _, d_head = k.shape
            shape = (batch_size, n_heads, self.max_seq_len, d_head)
            k_buf = torch.empty(shape, dtype=k.dtype, device=k.device)
            v_buf = torch.empty(shape, dtype=v.dtype, device=v.device)
            self._keys[layer_idx] = k_buf
            self._values[layer_idx] = v_buf

        k_buf[:, :, self._start_pos : self._end_pos, :] = k
        v_buf[:, :, self._start_pos : self._end_pos, :] = v
        return k_buf[:, :, : self._end_pos, :], v_buf[:, :, : self._end_pos, :]

    def commit(self) -> None:
        """Advance `seq_len` to `end_pos` once all layers have updated."""
        self._seq_len = self._end_pos


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention using `F.scaled_dot_product_attention`.

    Why `is_causal=True` with no padding mask on right-padded batches:
    In a right-padded `Batch`, real tokens sit at indices `0 .. length - 1` and
    `PAD` tokens sit at `>= length`. Under a causal mask, no position can attend
    to a later position, so real tokens never attend to `PAD` tokens and their
    representations are unaffected by padding. When decoding from a `LeftPaddedBatch`
    or incremental `KVCache`, callers pass `attn_mask` / `kv_cache` instead.
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention to `x` of shape `(B, T, d_model)`."""
        batch_size, seq_len, d_model = x.shape
        head_shape = (batch_size, seq_len, self._n_heads, self._d_head)

        q = self.q_proj(x).view(head_shape).transpose(1, 2)
        k = self.k_proj(x).view(head_shape).transpose(1, 2)
        v = self.v_proj(x).view(head_shape).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        use_causal = attn_mask is None and (kv_cache is None or kv_cache.start_pos == 0)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=use_causal
        )
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run one Pre-LN attention + FFN block on `x` of shape `(B, T, d_model)`."""
        x = x + self.attn(
            self.attn_norm(x),
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x
