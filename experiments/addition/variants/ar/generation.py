"""Free-running greedy generation (`rollout_ar`) for the `ar` variant of `addition`."""

import torch

from experiments.addition.inputs import StepOrder, validate_problem
from experiments.addition.variants.ar.inputs import build_ar_example
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context
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
    dtype: str | torch.dtype | None = None,
) -> list[Token]:
    """Run free-running greedy `ar` generation on `(alphabet, a, b)` until `[EOS]` or length cap.

    Starts with an empty trace after the third `[SEP]` (`input_ids` only, with no
    side-channel `token_type_ids` or `position_ids`), appends greedy `argmax`
    tokens from `SubspaceLMHead` up to
    `(4 if use_query_tokens else 2) * (max(len(a), len(b)) + 1) + 1` tokens
    (`spec/variants/ar.md` §3), and stops as soon as `SpecialToken.EOS` is emitted
    (including `EOS` as the final element of the returned list when emitted).
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(alphabet, a, b)
    _ = StepOrder(step_order)
    tokens_per_col = 4 if use_query_tokens else 2
    max_tokens = tokens_per_col * (max(len(a_tokens), len(b_tokens)) + 1) + 1

    device = model.device
    amp_ctx = autocast_context(device, dtype)
    model.eval()
    trace: list[Token] = []
    emitted: list[Token] = []

    with amp_ctx:
        for _ in range(max_tokens):
            example = build_ar_example(alphabet_tokens, a_tokens, b_tokens, trace)
            batch = collate([example]).to(device)
            logits = model(batch.input_ids, batch.lengths)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError(
                    "rollout_ar requires a Transformer with SubspaceLMHead"
                )
            next_tok = int(logits[0, -1, :].argmax(dim=-1).item())
            emitted.append(next_tok)
            if next_tok == SpecialToken.EOS:
                break
            trace.append(next_tok)

    return emitted
