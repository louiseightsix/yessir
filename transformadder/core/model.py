"""Modular TransformAdder Transformer trunk combining embeddings, blocks, and output heads.
Pure PyTorch implementation with zero external library dependencies.
"""

import math

import torch
import torch.nn as nn

from transformadder.core.blocks import TransformerBlock
from transformadder.core.constants import D_FFN, D_MODEL, N_HEADS, N_LAYERS
from transformadder.core.embedding import SubspaceEmbedding
from transformadder.core.heads import MultiHeadOutput, TiedIdentityHead
from transformadder.core.sequences import Seq, collate


class TransformAdder(nn.Module):
    """TransformAdder model trunk supporting both State Machine and Autoregressive paradigms."""

    def __init__(
        self,
        tier: int = 1,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ffn: int = D_FFN,
        head_type: str = "single_head",
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.tier = tier
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.head_type = head_type
        self.ffn_type = ffn_type

        # 1. Subspace embedding
        self.embedding = SubspaceEmbedding(tier=tier, d_model=d_model)

        # 2. Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, ffn_type=ffn_type)
                for _ in range(n_layers)
            ]
        )

        # 3. Output head
        if head_type == "single_head":
            self.head = TiedIdentityHead(d_model=d_model)
        elif head_type == "multi_head":
            self.head = MultiHeadOutput(d_model=d_model)
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'single_head' or 'multi_head'.")

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """N(0, 0.02^2) for all linear weights; residual output projections (out_proj, w_down)
        scaled by 1 / sqrt(2 * n_layers). Role embedding rows are unit norm."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual_scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            layer.attn.out_proj.weight.mul_(residual_scale)
            layer.ffn.w_down.weight.mul_(residual_scale)
        self.embedding.reset_role_embedding()

    def trunk(
        self,
        token_ids: torch.Tensor,
        role_ids: torch.Tensor | None = None,
        sig_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embedding and transformer blocks: [B, T] -> hidden states [B, T, 512]."""
        x = self.embedding(token_ids, role_ids=role_ids, sig_ids=sig_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

    def forward(
        self,
        token_ids: torch.Tensor,
        role_ids: torch.Tensor | None = None,
        sig_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tied-slice logits [B, T, 256] (SM-Seq, SM-SeqRev, AR).

        role_ids is required for Tiers 1 and 4, sig_ids for Tier 1; both are None for Tiers 2, 3 and 5. attn_mask=None selects
        strict causal attention; a supplied mask must be complete (see blocks.prefix_lm_mask).
        """
        if self.head_type != "single_head":
            raise ValueError("forward() is for single_head models; use forward_sm_mh().")
        return self.head(self.trunk(token_ids, role_ids, sig_ids, attn_mask))

    def forward_sm_mh(
        self,
        token_ids: torch.Tensor,
        role_ids: torch.Tensor | None,
        sig_ids: torch.Tensor | None,
        lengths: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(logits_sum [B, 256], logits_carry [B, 256], logits_flag [B, 2]), read at the true
        sequence tail, index lengths - 1 (the [S] query)."""
        if self.head_type != "multi_head":
            raise ValueError("forward_sm_mh() is for multi_head models.")
        return self.head(self.trunk(token_ids, role_ids, sig_ids, attn_mask), lengths)

    # Batch-of-one conveniences for the harnesses
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def next_logits(self, seq: Seq) -> torch.Tensor:
        """Tied-slice logits at the final position of a single sequence: [256]."""
        batch = collate([seq], device=self._device())
        return self.forward(batch["token_ids"], batch["role_ids"], batch["sig_ids"])[0, -1, :]

    def forward_sm_mh_single(self, seq: Seq) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SM-MH logits for a single sequence: ([256], [256], [2])."""
        batch = collate([seq], device=self._device())
        logits = self.forward_sm_mh(
            batch["token_ids"], batch["role_ids"], batch["sig_ids"], batch["lengths"]
        )
        return tuple(l[0] for l in logits)
