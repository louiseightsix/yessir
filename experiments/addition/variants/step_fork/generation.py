"""Free-running greedy rollout (`rollout_step_fork`) for the `step_fork` variant of `addition`."""

import torch

from experiments.addition.inputs import EmbeddingMode, validate_problem
from experiments.addition.variants.step_fork.inputs import build_step_fork_examples
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def rollout_step_fork(
    model: Transformer,
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[tuple[Token, Token, Token]]:
    """Run free-running greedy `step_fork` execution step by step on `(alphabet, a, b)`.

    Starts at `k = 0` with `carry = alphabet[0]` (`s_0`) and `output = []`. At each
    column step `k` (up to `max(len(a), len(b)) + 1` steps; `spec/variants/step_fork.md` §3),
    collates the 3 branched queries (`[S]`, `[C]`, `[F]`) into a 3-row batch, decodes
    `(o_k, c_{k+1}, flag_k)` at the final token (`-1`) of each row, and stops as soon
    as `flag_k == SpecialToken.FLAG_FINISHED`.
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(alphabet, a, b)
    max_columns = max(len(a_tokens), len(b_tokens)) + 1

    model.eval()
    device = model.device
    carry: Token = alphabet_tokens[0]
    output: list[Token] = []
    emitted_steps: list[tuple[Token, Token, Token]] = []

    for _ in range(max_columns):
        branches = build_step_fork_examples(
            alphabet_tokens,
            a_tokens,
            b_tokens,
            carry,
            output,
            embedding_mode=embedding_mode,
        )
        batch = collate(branches).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "rollout_step_fork requires a Transformer with SubspaceLMHead"
            )
        # All 3 branches share identical sequence length, so `-1` is the query token.
        preds = logits[:, -1, :].argmax(dim=-1).tolist()
        o_k, c_next, flag_k = int(preds[0]), int(preds[1]), int(preds[2])
        emitted_steps.append((o_k, c_next, flag_k))

        if flag_k == SpecialToken.FLAG_FINISHED:
            break
        carry = c_next
        output.append(o_k)

    return emitted_steps
