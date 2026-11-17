"""Free-running greedy generation (`rollout_ar`) for the `ar` variant of `addition`."""

import torch

from experiments.addition.inputs import EmbeddingMode, StepOrder, validate_problem
from experiments.addition.variants.ar.inputs import build_ar_example
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def rollout_ar(
    model: Transformer,
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[Token]:
    """Run free-running greedy `ar` generation on `(alphabet, a, b)` until `[EOS]` or length cap.

    Starts with an empty trace after the third `[SEP]`, appends greedy `argmax`
    tokens from `SubspaceLMHead` up to
    `(4 if use_query_tokens else 2) * (max(len(a), len(b)) + 1) + 1` tokens
    (`spec/variants/ar.md` §3), and stops as soon as `SpecialToken.EOS` is emitted
    (including `EOS` as the final element of the returned list when emitted).
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(alphabet, a, b)
    order = StepOrder(step_order)
    tokens_per_col = 4 if use_query_tokens else 2
    max_tokens = tokens_per_col * (max(len(a_tokens), len(b_tokens)) + 1) + 1

    model.eval()
    device = model.device
    trace: list[Token] = []
    emitted: list[Token] = []

    for _ in range(max_tokens):
        example = build_ar_example(
            alphabet_tokens,
            a_tokens,
            b_tokens,
            trace,
            step_order=order,
            use_query_tokens=use_query_tokens,
            embedding_mode=embedding_mode,
        )
        batch = collate([example]).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("rollout_ar requires a Transformer with SubspaceLMHead")
        next_tok = int(logits[0, -1, :].argmax(dim=-1).item())
        emitted.append(next_tok)
        if next_tok == SpecialToken.EOS:
            break
        trace.append(next_tok)

    return emitted
