"""State-machine execution harness (`run_sm_mh`) for the `sir_yes_sir` experiment."""

import torch

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    validate_problem,
)
from experiments.sir_yes_sir.variants.sm_mh.inputs import build_sm_example
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def run_sm_mh(
    model: Transformer,
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[tuple[Token, Token, Token]]:
    """Run the `sm_mh` state-machine harness step by step on `(alphabet, x)`.

    At each step `k = 0 .. len(x) - 1`, builds the state-machine `Example` using
    the model's own previously emitted primary outputs `output_history = [o_0, ..., o_{k-1}]`,
    runs one forward pass through `model`, and decodes `(o_k, w_k, flag_k)` by
    greedy `argmax` across the three heads of `MultiTaskHead`.

    Stops and returns the list of emitted `(o_k, w_k, flag_k)` step tuples as soon as:
    1. the model emits `flag_k == SpecialToken.FLAG_FINISHED`,
    2. `len(x)` steps have been executed (the maximum legal step count for `build_sm_example`), or
    3. the model emits an out-of-alphabet primary symbol `o_k not in alphabet`
       (which cannot be fed into `build_sm_example` on the next step).
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    alphabet_set = set(alphabet_tokens)

    model.eval()
    output_history: list[Token] = []
    emitted_steps: list[tuple[Token, Token, Token]] = []

    for _ in range(len(x_tokens)):
        example = build_sm_example(
            alphabet_tokens,
            x_tokens,
            output_history,
            embedding_mode=embedding_mode,
        )
        batch = collate([example])
        logits_o, logits_w, logits_flag = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )

        o_k = int(logits_o[0].argmax(dim=-1).item())
        w_k = int(logits_w[0].argmax(dim=-1).item())
        flag_k = int(logits_flag[0].argmax(dim=-1).item())
        emitted_steps.append((o_k, w_k, flag_k))

        if flag_k == SpecialToken.FLAG_FINISHED or o_k not in alphabet_set:
            break
        output_history.append(o_k)

    return emitted_steps
