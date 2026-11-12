"""Output heads for mapping final Transformer hidden states to task logits."""

from collections.abc import Sequence

import torch
from torch import nn

from yessir.blocks import RMSNorm


class MultiTaskHead(nn.Module):
    """Parallel linear classification heads reading at the last unpadded token (`lengths - 1`).

    In a right-padded `Batch`, example `b` has real tokens at indices
    `0 .. lengths[b] - 1` and `PAD` tokens at `>= lengths[b]`. Rather than reading
    `h[:, -1]` (which is a `PAD` position whenever examples in the batch have
    different lengths), `MultiTaskHead` gathers `h[b, lengths[b] - 1]` for each
    example, applies a final `RMSNorm`, and projects that single hidden state
    through each head in `self.heads`.
    """

    def __init__(self, d_model: int, num_classes: Sequence[int]) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if not num_classes:
            raise ValueError("num_classes must contain at least one head")
        for idx, n_cls in enumerate(num_classes):
            if n_cls <= 0:
                raise ValueError(f"num_classes[{idx}] must be > 0, got {n_cls}")

        self.norm = RMSNorm(d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, n_cls, bias=False) for n_cls in num_classes]
        )

    def forward(
        self, h: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Project the last unpadded hidden state of each sequence into per-head logits.

        Args:
            h: Hidden states of shape `(B, T, d_model)`.
            lengths: Unpadded sequence lengths of shape `(B,)` with values in `1..T`.

        Returns:
            Tuple of `(B, num_classes_i)` logit tensors, one per head.
        """
        batch_size = h.size(0)
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"lengths shape {tuple(lengths.shape)} must match batch size ({batch_size},)"
            )

        batch_indices = torch.arange(batch_size, device=h.device)
        # Normalising after indexing `lengths - 1` is equivalent to normalising all
        # T positions first (RMSNorm is per-position), at 1/T the compute cost.
        h_last = self.norm(h[batch_indices, lengths - 1])
        return tuple(head(h_last) for head in self.heads)
