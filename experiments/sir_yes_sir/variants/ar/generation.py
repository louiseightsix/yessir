"""Free-running greedy autoregressive generation (`greedy_generate_ar`) for the `ar` variant of `sir_yes_sir`."""

import torch

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    build_example,
    validate_problem,
)
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def greedy_generate_ar(
    model: Transformer,
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    dtype: str | torch.dtype | None = None,
) -> list[Token]:
    """Run free-running greedy autoregressive generation on `(alphabet, x)`.

    Starts with `output = ()` after the second `[SEP]` (`input_ids` only, with no
    side-channel `token_type_ids` or `position_ids`) and greedily appends one
    `argmax` token per forward pass from `SubspaceLMHead` (up to `len(x) + 1`
    tokens), stopping as soon as `SpecialToken.EOS` is emitted
    (`experiments/sir_yes_sir/spec/variants/ar.md` §2).

    Args:
        model: `Transformer` configured with `SubspaceLMHead`.
        alphabet: Ordered problem alphabet of `B >= 2` distinct byte symbols.
        x: Input sequence of `L >= 1` symbols drawn from `alphabet`.
        dtype: Compute dtype (`'auto'` / `None`, `'float32'`, or `'bfloat16'`).

    Returns:
        List of generated tokens (including the terminating `SpecialToken.EOS` if emitted).
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    max_tokens = len(x_tokens) + 1

    device = model.device
    amp_ctx = autocast_context(device, dtype)
    model.eval()
    output: list[Token] = []

    with amp_ctx:
        for _ in range(max_tokens):
            example = build_example(
                alphabet_tokens,
                x_tokens,
                output,
                embedding_mode=EmbeddingMode.NOPE,
            )
            batch = collate([example]).to(device)
            logits = model(batch.input_ids, batch.lengths)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError(
                    "greedy_generate_ar requires a Transformer with SubspaceLMHead"
                )
            next_token = int(logits[0, -1].argmax(dim=-1).item())
            output.append(next_token)

            if next_token == SpecialToken.EOS:
                break

    return output
