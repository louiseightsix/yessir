"""Pre-LN Transformer model (`Transformer`)."""

import math
from collections.abc import Sequence

import torch
from torch import nn

from yessir.blocks import RMSNorm, TransformerBlock
from yessir.embedding import SubspaceEmbedding
from yessir.heads import HeadType, ParallelClassificationHead, SubspaceLMHead

_INIT_STD = 0.02

# A standard 2-matrix FFN at the default d_model = 512 uses d_ff = 4 * d_model = 2048.
# SwiGLU uses 3 matrices (gate_proj, up_proj, down_proj), so matching that parameter
# count takes (2 / 3) * 4 * 512 ≈ 1365.3 hidden units; rounding up to a multiple of 32
# gives 43 * 32 = 1376. Callers that change d_model (via d_scratch / d_position /
# use_token_type) should pass d_ff explicitly if they want d_ff to scale with it.
_DEFAULT_D_FF = 1376


class Transformer(nn.Module):
    """Pre-LN Transformer combining `SubspaceEmbedding`, `TransformerBlock`s, and an output head.

    Output head selection (`head_type`):
    - `HeadType.PARALLEL_CLASSIFICATION`: requires `num_classes` (e.g. `(256, 256)`) and
      builds a `ParallelClassificationHead` that projects either the final unpadded
      position `lengths - 1` (when `lengths` is passed) or all sequence positions
      `(B, T, num_classes_i)` (when `lengths=None`) into per-head logits.
    - `HeadType.SUBSPACE_LM`: requires `num_classes=None` and builds a parameter-free
      `SubspaceLMHead` that reads the `0..255` symbol subspace across all sequence
      positions `(B, T, 256)`.

    Weight initialisation (`reset_parameters`):
    - Every `nn.Linear` weight is drawn from `N(0, 0.02^2)` (GPT-2 / LLaMA convention).
    - Each `TransformerBlock` adds two residual branches (`attn` and `ffn`) into the
      residual stream, giving `2 * n_layers` additions across the network. To keep
      the variance accumulated along the residual stream independent of depth, the
      output projection of each residual branch (`attn.out_proj` and `ffn.down_proj`)
      is scaled by `1 / sqrt(2 * n_layers)` (Radford et al., 2019).
    """

    def __init__(
        self,
        *,
        head_type: HeadType,
        num_classes: Sequence[int] | None = None,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = _DEFAULT_D_FF,
        use_token_type: bool = True,
        d_position: int = 32,
        d_scratch: int = 208,
    ) -> None:
        super().__init__()
        head_type = HeadType(head_type)
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if head_type == HeadType.PARALLEL_CLASSIFICATION and num_classes is None:
            raise ValueError(
                "num_classes is required when head_type is 'parallel_classification'"
            )
        if head_type == HeadType.SUBSPACE_LM and num_classes is not None:
            raise ValueError("num_classes must be None when head_type is 'subspace_lm'")

        self.embedding = SubspaceEmbedding(
            use_token_type=use_token_type,
            d_position=d_position,
            d_scratch=d_scratch,
        )
        self.d_model = self.embedding.d_model

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                )
                for _ in range(n_layers)
            ]
        )
        self.head: ParallelClassificationHead | SubspaceLMHead
        if head_type == HeadType.PARALLEL_CLASSIFICATION:
            assert num_classes is not None
            self.head = ParallelClassificationHead(
                d_model=self.d_model, num_classes=num_classes
            )
        else:
            self.head = SubspaceLMHead(d_model=self.d_model)

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        """The `torch.device` this model's parameters reside on."""
        return next(self.parameters()).device

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialise linear weights from `N(0, 0.02^2)` with `1 / sqrt(2 * n_layers)` residual scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
            elif isinstance(module, RMSNorm) and module.weight is not None:
                nn.init.ones_(module.weight)

        residual_scale = 1.0 / math.sqrt(2 * len(self.layers))
        for layer in self.layers:
            assert isinstance(layer, TransformerBlock)
            layer.attn.out_proj.weight.mul_(residual_scale)
            layer.ffn.down_proj.weight.mul_(residual_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run embedding, transformer blocks, and output head on a padded batch."""
        x = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        for layer in self.layers:
            x = layer(x)
        if isinstance(self.head, SubspaceLMHead):
            return self.head(x)
        return self.head(x, lengths)
