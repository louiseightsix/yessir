"""Output heads (`HeadType`, `MultiTaskHead`, `TiedIdentityHead`) for mapping final Transformer hidden states to task logits."""

from collections.abc import Sequence
from enum import StrEnum, unique

import torch
from torch import nn

from yessir.blocks import RMSNorm
from yessir.vocab import VOCAB_SIZE


@unique
class HeadType(StrEnum):
    """Output head architecture for `Transformer`."""

    MULTITASK = "multitask"
    TIED_IDENTITY = "tied_identity"


class TiedIdentityHead(nn.Module):
    """Parameter-free tied output head reading the `0..255` symbol subspace at all positions.

    Because `SubspaceEmbedding` places the 256-d one-hot token representation in
    coordinates `0..255` of the residual stream, applying a gain-free `RMSNorm` and
    slicing `[:, :, :VOCAB_SIZE]` computes the dot product of the normalised hidden
    state with each token's unit-vector embedding (up to the shared `sqrt(d_model)`
    RMS scale factor), with zero trainable parameters.

    Initialisation note (self-copy bias vs. `MultiTaskHead`):
    At step 0 of training, residual branch output projections are small, so the final
    hidden state at position `t` is still dominated by `SubspaceEmbedding`'s input
    one-hot vector (`+1` at coordinate `input_ids[b, t]` and `0` elsewhere in `0..255`).
    As a result, `TiedIdentityHead` initially assigns its largest logit to repeating
    the current position's input token (just like standard Pre-LN weight-tied
    transformers `norm(h) @ W_E^T`), and must learn through the residual blocks to
    suppress or outweigh the input token coordinate when predicting the next token.
    `MultiTaskHead`, by contrast, projects through freshly initialised linear weights
    and starts with near-uniform logits (`loss ≈ ln(256)` per head).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < VOCAB_SIZE:
            raise ValueError(
                f"d_model must be >= VOCAB_SIZE ({VOCAB_SIZE}) for TiedIdentityHead, got {d_model}"
            )
        self.norm = RMSNorm(d_model, gain=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalise `hidden_states` `(B, T, d_model)` and slice `(B, T, VOCAB_SIZE)` logits."""
        del lengths
        return self.norm(hidden_states)[:, :, :VOCAB_SIZE]


class MultiTaskHead(nn.Module):
    """Parallel linear classification heads reading at the last unpadded token (`lengths - 1`).

    In a right-padded `Batch`, example `b` has real tokens at indices
    `0 .. lengths[b] - 1` and `PAD` tokens at `>= lengths[b]`. Rather than reading
    `hidden_states[:, -1]` (which is a `PAD` position whenever examples in the batch
    have different lengths), `MultiTaskHead` gathers `hidden_states[b, lengths[b] - 1]`
    for each example, applies a final `RMSNorm`, and projects that single hidden state
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
        self, hidden_states: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Project the last unpadded hidden state of each sequence into per-head logits.

        Assumes `lengths` values lie in `1..T` (enforced by `Example` and `collate`;
        `0` is not checked—that would force a device sync—and would read index `-1`,
        the last position).

        Args:
            hidden_states: Hidden states of shape `(B, T, d_model)`.
            lengths: Unpadded sequence lengths of shape `(B,)` with values in `1..T`.

        Returns:
            Tuple of `(B, num_classes_i)` logit tensors, one per head.
        """
        batch_size = hidden_states.size(0)
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"lengths shape {tuple(lengths.shape)} must match batch size ({batch_size},)"
            )

        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        # Normalising after indexing `lengths - 1` is equivalent to normalising all
        # T positions first (RMSNorm is per-position), at 1/T the compute cost.
        h_last = self.norm(hidden_states[batch_indices, lengths - 1])
        return tuple(head(h_last) for head in self.heads)
