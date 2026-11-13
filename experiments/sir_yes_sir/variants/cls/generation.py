"""Free-running greedy parallel-classification generation (`greedy_generate_cls`) for the `cls` variant of `sir_yes_sir`."""

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
def greedy_generate_cls(
    model: Transformer,
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[tuple[Token, Token]]:
    """Run free-running `cls` greedy generation step by step on `(alphabet, x)`.

    At each step `k = 0 .. len(x) - 1`, builds the shared `Example` using the
    model's own previously generated outputs `output = [o_0, ..., o_{k-1}]`, runs
    one forward pass through `model` (reading the final position `lengths - 1`),
    and decodes `(o_k, flag_k)` by greedy `argmax` across the two heads of
    `ParallelClassificationHead` (`experiments/sir_yes_sir/spec/variants/cls.md` §2).

    Stops and returns the list of generated `(o_k, flag_k)` step tuples as soon as
    `flag_k == SpecialToken.FLAG_FINISHED` or after `len(x)` steps.

    Args:
        model: `Transformer` configured with a 2-head `ParallelClassificationHead`.
        alphabet: Ordered problem alphabet of `B >= 2` distinct byte symbols.
        x: Input sequence of `L >= 1` symbols drawn from `alphabet`.
        embedding_mode: Which of `token_type_ids` and `position_ids` to attach.

    Returns:
        List of generated `(o_k, flag_k)` step tuples.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)

    model.eval()
    device = model.device
    output: list[Token] = []
    steps: list[tuple[Token, Token]] = []

    for _ in range(len(x_tokens)):
        example = build_example(
            alphabet_tokens,
            x_tokens,
            output,
            embedding_mode=embedding_mode,
        )
        batch = collate([example]).to(device)
        logits_o, logits_flag = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )

        o_k = int(logits_o[0].argmax(dim=-1).item())
        flag_k = int(logits_flag[0].argmax(dim=-1).item())
        steps.append((o_k, flag_k))

        if flag_k == SpecialToken.FLAG_FINISHED:
            break
        output.append(o_k)

    return steps
