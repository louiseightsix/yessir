"""Pre-LN Transformer model (`Transformer`)."""

import math
from collections.abc import Sequence

import torch
from torch import nn

from yessir.blocks import RMSNorm, TransformerBlock
from yessir.embedding import SubspaceEmbedding
from yessir.heads import MultiTaskHead

_INIT_STD = 0.02

# A standard 2-matrix FFN at the default d_model = 512 uses d_ffn = 4 * d_model = 2048.
# SwiGLU uses 3 matrices (w_gate, w_up, w_down), so matching that parameter count
# takes (2 / 3) * 4 * 512 ≈ 1365.3 hidden units; rounding up to a multiple of 32
# gives 43 * 32 = 1376. Callers that change d_model (via d_scratch / d_position /
# use_token_type) should pass d_ffn explicitly if they want d_ffn to scale with it.
_DEFAULT_D_FFN = 1376


class Transformer(nn.Module):
    """Pre-LN Transformer combining `SubspaceEmbedding`, `TransformerBlock`s, and `MultiTaskHead`.

    Weight initialisation (`reset_parameters`):
    - Every `nn.Linear` weight is drawn from `N(0, 0.02^2)` (GPT-2 / LLaMA convention).
    - Each `TransformerBlock` adds two residual branches (`attn` and `ffn`) into the
      residual stream, giving `2 * n_layers` additions across the network. To keep
      the variance accumulated along the residual stream independent of depth, the
      output projection of each residual branch (`attn.out_proj` and `ffn.w_down`)
      is scaled by `1 / sqrt(2 * n_layers)` (Radford et al., 2019).
    """

    def __init__(
        self,
        *,
        num_classes: Sequence[int],
        n_layers: int = 6,
        n_heads: int = 8,
        d_ffn: int = _DEFAULT_D_FFN,
        use_token_type: bool = True,
        d_position: int = 32,
        d_scratch: int = 208,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")

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
                    d_ffn=d_ffn,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = MultiTaskHead(d_model=self.d_model, num_classes=num_classes)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialise linear weights from `N(0, 0.02^2)` with `1 / sqrt(2 * n_layers)` residual scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        residual_scale = 1.0 / math.sqrt(2 * len(self.layers))
        for layer in self.layers:
            assert isinstance(layer, TransformerBlock)
            layer.attn.out_proj.weight.mul_(residual_scale)
            layer.ffn.w_down.weight.mul_(residual_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run embedding, transformer blocks, and multi-task head on a padded batch."""
        x = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        for layer in self.layers:
            x = layer(x)
        return self.head(x, lengths)
