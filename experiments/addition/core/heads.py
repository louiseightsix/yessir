"""Output head topologies for the TransformAdder model.
- TiedIdentityHead: Zero-parameter copy slice of the symbol dims (SM-Seq, SM-SeqRev, SM-Fork, AR).
- MultiHeadOutput: Parallel linear projections for sum, carry, and flag (SM-MH).
"""

from enum import Enum

import torch
import torch.nn as nn

from yessir.blocks import RMSNorm
from experiments.addition.core.constants import D_MODEL, D_SYMBOL


class Channel(str, Enum):
    SEQUENCE = "sequence"
    WEIGHT = "weight"


class TiedIdentityHead(nn.Module):
    """Parameter-free tied copy slice.

    Applies the gain-free final norm and extracts dims 0..255 (the symbol identity subspace):
      y_t = RMSNorm_0(h_t)[:256]
    Contains zero trainable weights.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.norm = RMSNorm(d_model, gain=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, 512] -> logits [B, T, 256]."""
        return self.norm(h)[:, :, :D_SYMBOL]


class MultiHeadOutput(nn.Module):
    """Multi-head parallel projection module for Paradigm A1 (SM-MH).

    Branches off the hidden state at the [S] query position (the true, unpadded sequence tail):
      logits_sum: [B, 256]
      logits_carry: [B, 256]
      logits_flag: [B, 2] (0: FLAG_CONTINUE, 1: FLAG_FINISHED)
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.head_sum = nn.Linear(d_model, D_SYMBOL, bias=False)
        self.head_carry = nn.Linear(d_model, D_SYMBOL, bias=False)
        self.head_flag = nn.Linear(d_model, 2, bias=False)

    def forward(
        self, h: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: [B, T, 512]; lengths: [B] true (unpadded) sequence lengths.

        Reads index lengths - 1, never h[:, -1], which in a right-padded batch is a PAD position.
        """
        batch = torch.arange(h.size(0), device=h.device)
        h_tail = self.norm(h[batch, lengths - 1, :])
        return self.head_sum(h_tail), self.head_carry(h_tail), self.head_flag(h_tail)
