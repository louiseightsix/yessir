"""Transformer building blocks: RMSNorm, SwiGLU FFN, and Multi-Head Attention via SDPA.
Pure PyTorch implementation with zero external framework dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Pre-LN).

    With gain=False the norm is parameter-free (RMSNorm_0), as required for the final norm
    feeding the tied copy slice.
    """

    def __init__(self, dim: int, eps: float = 1e-6, gain: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if gain else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x if self.weight is None else x * self.weight


class SwiGLUFeedForward(nn.Module):
    """Gated bilinear feed-forward layer implementing carry and modulo non-linearities."""

    def __init__(self, d_model: int = 512, d_ffn: int = 1376):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GELUFeedForward(nn.Module):
    """Standard 2-layer MLP with GELU activation for ablation comparison."""

    def __init__(self, d_model: int = 512, d_ffn: int = 2048):
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_up(x)))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention utilising PyTorch SDPA.

    Mask contract: attn_mask=None selects the built-in strictly causal mask, which is sufficient
    for every right-padded batch. A supplied attn_mask REPLACES the causal mask entirely and must
    itself encode causality (used only for the Tier-1 Prefix-LM ablation, see prefix_lm_mask).
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # attn_mask: optional bool [B, 1, T, T], True = may attend. Must be a COMPLETE mask.
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


def prefix_lm_mask(prefix_lens: torch.Tensor, T: int) -> torch.Tensor:
    """Complete Prefix-LM mask [B, 1, T, T]: causal everywhere, plus full bidirectional
    visibility inside each sequence's static prefix ([BOS], A, a, b). Tier 1 only: without
    positional inputs a bidirectional prefix is an unordered bag and the axiom's order is lost.
    """
    idx = torch.arange(T, device=prefix_lens.device)
    causal = idx[None, :, None] >= idx[None, None, :]      # [1, T, T]
    in_prefix = idx[None, :] < prefix_lens[:, None]        # [B, T]
    bidir = in_prefix[:, :, None] & in_prefix[:, None, :]  # [B, T, T]
    return (causal | bidir)[:, None, :, :]


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block combining MultiHeadSelfAttention and FFN (SwiGLU or GELU)."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ffn: int = 1376, ffn_type: str = "swiglu"):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        if ffn_type == "gelu":
            self.ffn = GELUFeedForward(d_model, d_ffn if d_ffn != 1376 else 2048)
        else:
            self.ffn = SwiGLUFeedForward(d_model, d_ffn)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
