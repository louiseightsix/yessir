"""Autoregressive execution harness (`parse_ar_trace`, `rollout_ar`) for the `ar` variant of `sir_yes_sir`."""

import torch

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    validate_problem,
)
from experiments.sir_yes_sir.variants.ar.inputs import (
    StepOrder,
    build_ar_example,
)
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def parse_ar_trace(
    trace: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.OUT_FIRST,
) -> list[tuple[Token, Token, Token]]:
    """Parse a flat autoregressive `trace` into canonical `(o_k, w_k, flag_k)` step tuples.

    Only complete 3-token step chunks are returned; if `step_order == StepOrder.WRAP_FIRST`,
    each `(w_k, o_k, flag_k)` triplet is reordered to `(o_k, w_k, flag_k)` so the returned
    tuples always match `reference_step`.

    Args:
        trace: Flat sequence of generated output tokens.
        step_order: Within-step emission order (`OUT_FIRST` or `WRAP_FIRST`).

    Returns:
        List of canonical `(o_k, w_k, flag_k)` tuples for all complete 3-token steps.
    """
    step_order = StepOrder(step_order)
    tokens = tuple(int(t) for t in trace)
    steps: list[tuple[Token, Token, Token]] = []
    for i in range(0, len(tokens) - 2, 3):
        first, second, flag_k = tokens[i], tokens[i + 1], tokens[i + 2]
        if step_order == StepOrder.OUT_FIRST:
            steps.append((first, second, flag_k))
        else:
            steps.append((second, first, flag_k))
    return steps


@torch.no_grad()
def rollout_ar(
    model: Transformer,
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.OUT_FIRST,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[tuple[Token, Token, Token]]:
    """Run free-running greedy autoregressive rollout on `(alphabet, x)`.

    Starts from an empty trace after the second `[SEP]` and greedily appends one
    token per forward pass from `model` (up to `3 * len(x)` tokens), stopping as
    soon as a 3-token step ends with `SpecialToken.FLAG_FINISHED`.

    Args:
        model: `Transformer` configured with `SubspaceLMHead`.
        alphabet: Ordered problem alphabet of `B >= 2` distinct byte symbols.
        x: Input sequence of `L >= 1` symbols drawn from `alphabet`.
        step_order: Within-step emission order (`OUT_FIRST` or `WRAP_FIRST`).
        embedding_mode: Which of `token_type_ids` and `position_ids` to attach.

    Returns:
        List of canonical `(o_k, w_k, flag_k)` step tuples parsed from the trace.
    """
    step_order = StepOrder(step_order)
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    max_tokens = 3 * len(x_tokens)

    model.eval()
    device = model.device
    trace: list[Token] = []

    for _ in range(max_tokens):
        example = build_ar_example(
            alphabet_tokens,
            x_tokens,
            trace,
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
        trace.append(next_token)

        if len(trace) % 3 == 0 and next_token == SpecialToken.FLAG_FINISHED:
            break

    return parse_ar_trace(trace, step_order=step_order)
