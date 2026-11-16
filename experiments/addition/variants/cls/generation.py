"""Free-running greedy rollout (`rollout_cls`) for the `cls` variant of `addition`."""

import torch

from experiments.addition.inputs import EmbeddingMode, validate_problem
from experiments.addition.variants.cls.inputs import build_cls_example
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@torch.no_grad()
def rollout_cls(
    model: Transformer,
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[tuple[Token, Token, Token]]:
    """Run free-running greedy `cls` execution step by step on `(alphabet, a, b)`.

    Starts at `k = 0` with `carry = alphabet[0]` (`s_0`) and `output = []`, and
    runs up to `max(len(a), len(b)) + 1` steps (`spec/variants/cls.md` §3),
    stopping as soon as `flag_k == SpecialToken.FLAG_FINISHED`.
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(alphabet, a, b)
    max_columns = max(len(a_tokens), len(b_tokens)) + 1

    model.eval()
    device = model.device
    carry: Token = alphabet_tokens[0]
    output: list[Token] = []
    emitted_steps: list[tuple[Token, Token, Token]] = []

    for _ in range(max_columns):
        example = build_cls_example(
            alphabet_tokens,
            a_tokens,
            b_tokens,
            carry,
            output,
            embedding_mode=embedding_mode,
        )
        batch = collate([example]).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, tuple) or len(logits) != 3:
            raise ValueError(
                "rollout_cls requires a Transformer with a 3-head ParallelClassificationHead"
            )
        logits_o, logits_c, logits_flag = logits
        o_k = int(logits_o[0].argmax(dim=-1).item())
        c_next = int(logits_c[0].argmax(dim=-1).item())
        flag_k = int(logits_flag[0].argmax(dim=-1).item())
        emitted_steps.append((o_k, c_next, flag_k))

        if flag_k == SpecialToken.FLAG_FINISHED:
            break
        carry = c_next
        output.append(o_k)

    return emitted_steps
