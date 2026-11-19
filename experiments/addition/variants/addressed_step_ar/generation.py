"""Free-running greedy rollout (`rollout_addressed_step_ar`) for the `addressed_step_ar` variant of `addition`."""

import torch

from experiments.addition.inputs import EmbeddingMode, StepOrder, validate_problem
from experiments.addition.variants.addressed_step_ar.inputs import (
    build_addressed_step_ar_example,
    unpack_step_order,
)
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def rollout_addressed_step_ar(
    model: Transformer,
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
    dtype: str | torch.dtype | None = None,
) -> list[tuple[Token, Token, Token]]:
    """Run free-running greedy `addressed_step_ar` execution step by step on `(alphabet, a, b)`.

    Starts at `k = 0` with `carry = alphabet[0]` (`s_0`) and `output = []`. At each
    column step `k` (up to `max(len(a), len(b)) + 1` steps; `spec/variants/addressed_step_ar.md` §3),
    runs 3 sequential greedy `argmax` passes (inserting `Q_0, Q_1, Q_2` when
    `use_query_tokens=True`) to decode `(t_0, t_1, flag_k)`, unpacks `(o_k, c_{k+1})`
    according to `step_order`, and stops as soon as `flag_k == SpecialToken.FLAG_FINISHED`.
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(
        alphabet, a, b, include_alphabet=include_alphabet
    )
    order = StepOrder(step_order)
    max_columns = max(len(a_tokens), len(b_tokens)) + 1

    device = model.device
    amp_ctx = autocast_context(device, dtype)
    model.eval()
    carry: Token = alphabet_tokens[0]
    output: list[Token] = []
    emitted_steps: list[tuple[Token, Token, Token]] = []

    with amp_ctx:
        for _ in range(max_columns):
            step_emitted: list[Token] = []
            for _ in range(3):
                example = build_addressed_step_ar_example(
                    alphabet_tokens,
                    a_tokens,
                    b_tokens,
                    carry,
                    output,
                    step_emitted,
                    step_order=order,
                    use_query_tokens=use_query_tokens,
                    include_alphabet=include_alphabet,
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
                    raise ValueError(
                        "rollout_addressed_step_ar requires a Transformer with SubspaceLMHead"
                    )
                next_tok = int(logits[0, -1, :].argmax(dim=-1).item())
                step_emitted.append(next_tok)

            t_0, t_1, flag_k = step_emitted
            o_k, c_next = unpack_step_order(t_0, t_1, order)
            emitted_steps.append((o_k, c_next, flag_k))

            if flag_k == SpecialToken.FLAG_FINISHED:
                break
            carry = c_next
            output.append(o_k)

    return emitted_steps
