"""Subspace embedding module mapping 1-byte tokens to 512-dimensional representations.
Dims 0..255: Fixed one-hot symbol basis.
Dims 256..511: 2D Split PE (Tier 1), Role + NoPE (Tier 4), or static zero-padded state memory (Tiers 2, 3, 5, 6 & 7).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformadder.core.constants import D_MODEL, D_POS, D_ROLE, D_SIG, NUM_ROLES, VOCAB_SIZE


class SubspaceEmbedding(nn.Module):
    """Subspace embedding module for TransformAdder.

    Maps byte tokens into R^512:
      x_t = [ e_{b_t} || p_t ]
    where e_{b_t} is the non-trainable one-hot byte vector, and p_t is the positional/state vector.
    In Tier 1 the three components (one-hot, role, significance) each have unit L2 norm so that
    none dominates the first RMSNorm.
    """

    def __init__(self, tier: int = 1, d_model: int = D_MODEL):
        super().__init__()
        assert d_model == D_MODEL, f"TransformAdder requires d_model={D_MODEL}, got {d_model}"
        assert tier in (1, 2, 3, 4, 5, 6, 7), f"Unknown tier: {tier}"
        self.tier = tier

        if tier in (1, 4):
            self.role_embed = nn.Embedding(NUM_ROLES, D_ROLE)
            self.reset_role_embedding()
        else:
            self.role_embed = None

        if tier == 1:
            # omega_i = 10000^(-2i / 192) for i in {0 .. 95}
            half_sig = D_SIG // 2
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_sig, dtype=torch.float32) / half_sig))
            self.register_buffer("inv_freq", inv_freq)
            # An unscaled 192-dim sinusoid has norm sqrt(96); rescale to unit norm
            self.sig_scale = 1.0 / math.sqrt(half_sig)

    @torch.no_grad()
    def reset_role_embedding(self):
        """Initialise role embedding rows to unit norm."""
        if self.role_embed is not None:
            nn.init.normal_(self.role_embed.weight)
            self.role_embed.weight.copy_(F.normalize(self.role_embed.weight, dim=-1))

    def _compute_sinusoidal(self, sig_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Unit-norm sinusoidal significance coordinates for Tier 1: shape [B, T, 192].

        Interleaved layout: e_sig(p)[2i] = sin(p * omega_i), e_sig(p)[2i+1] = cos(p * omega_i).
        """
        angles = sig_ids.unsqueeze(-1).float() * self.inv_freq  # [B, T, 96]
        sig = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)
        return (sig * self.sig_scale).to(dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        role_ids: torch.Tensor | None = None,
        sig_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward embedding pass.

        Args:
            token_ids: Byte token indices of shape [B, T], values in 0..255.
            role_ids: Role indices of shape [B, T] (required for Tier 1).
            sig_ids: Significance coordinates of shape [B, T] (required for Tier 1).

        Returns:
            Tensor of shape [B, T, 512].
        """
        # 1. Symbol Identity Subspace: one-hot byte representation in dims 0..255
        one_hot = F.one_hot(token_ids, num_classes=VOCAB_SIZE).float()

        # 2. Positional / State Subspace in dims 256..511
        if self.tier == 1:
            if role_ids is None or sig_ids is None:
                raise ValueError("Tier 1 embedding requires both role_ids and sig_ids tensors.")
            role_vec = self.role_embed(role_ids)  # [B, T, 64]
            sig_vec = self._compute_sinusoidal(sig_ids, dtype=one_hot.dtype)  # [B, T, 192]
            return torch.cat([one_hot, role_vec, sig_vec], dim=-1)

        if self.tier == 4:
            if role_ids is None:
                raise ValueError("Tier 4 embedding requires role_ids tensor.")
            role_vec = self.role_embed(role_ids)  # [B, T, 64]
            sig_zeros = torch.zeros(
                token_ids.shape[0], token_ids.shape[1], D_SIG,
                device=token_ids.device, dtype=one_hot.dtype
            )
            return torch.cat([one_hot, role_vec, sig_zeros], dim=-1)

        # Tiers 2, 3, 5: NoPE (pure zeros in positional subspace)
        return F.pad(one_hot, (0, D_POS))
