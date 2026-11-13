"""Autoregressive execution harness (`rollout_ar`) for the `ar` variant of `sir_yes_sir`."""

import torch

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    build_example,
    validate_problem,
)
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def rollout_ar(
    model: Transformer,
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[Token]:
    """Run free-running greedy autoregressive rollout on `(alphabet, x)`.

    Starts with `output = ()` after the second `[SEP]` and greedily appends one
    `argmax` token per forward pass from `SubspaceLMHead` (up to `len(x) + 1`
    tokens), stopping as soon as `SpecialToken.EOS` is emitted
    (`experiments/sir_yes_sir/spec/variants/ar.md` §2).

    Args:
        model: `Transformer` configured with `SubspaceLMHead`.
        alphabet: Ordered problem alphabet of `B >= 2` distinct byte symbols.
        x: Input sequence of `L >= 1` symbols drawn from `alphabet`.
        embedding_mode: Which of `token_type_ids` and `position_ids` to attach.

    Returns:
        List of emitted tokens (including the terminating `SpecialToken.EOS` if emitted).
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    max_tokens = len(x_tokens) + 1

    model.eval()
    device = model.device
    emitted: list[Token] = []

    for _ in range(max_tokens):
        example = build_example(
            alphabet_tokens,
            x_tokens,
            emitted,
            embedding_mode=embedding_mode,
        )
        batch = collate([example]).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, torch.Tensor)
        next_token = int(logits[0, -1].argmax(dim=-1).item())
        emitted.append(next_token)

        if next_token == SpecialToken.EOS:
            break

    return emitted
