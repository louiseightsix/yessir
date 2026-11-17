"""Free-running greedy generation (`rollout_addressed_ar`) and `[IDX]`-segment parsing (`parse_addressed_ar_trace`) for the `addressed_ar` variant of `addition`."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from experiments.addition.inputs import (
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    encode_address,
    step_query_tokens,
    unpack_step_order,
    validate_problem,
)
from experiments.addition.variants.addressed_ar.inputs import (
    build_addressed_ar_example,
)
from yessir.batch import collate
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@dataclass(frozen=True)
class AddressedTraceStep:
    """One parsed `[IDX] addr(k) ...` column entry from an `addressed_ar` trace."""

    address: tuple[Token, ...]
    sum_digit: Token
    carry_out: Token


def parse_addressed_ar_trace(
    emitted: Sequence[Token],
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
) -> tuple[list[AddressedTraceStep], bool]:
    """Parse an emitted `addressed_ar` token stream (`[IDX] addr(k) ... [EOS]`) into column predictions.

    Returns `(steps, is_well_formed)`, where `is_well_formed` is `True` iff `emitted`
    terminates at a single trailing `[EOS]` preceded by one or more well-formed
    `[IDX] addr(k) ...` segments.
    """
    order = StepOrder(step_order)
    q_0, q_1, _ = step_query_tokens(order)
    idx_tok = int(AdditionToken.IDX)
    eos = int(SpecialToken.EOS)
    body_len = 4 if use_query_tokens else 2

    tokens = [int(t) for t in emitted]
    if not tokens or tokens[-1] != eos or eos in tokens[:-1]:
        halted_cleanly = False
        body_tokens = tokens[: tokens.index(eos)] if eos in tokens else tokens
    else:
        halted_cleanly = True
        body_tokens = tokens[:-1]

    if not body_tokens or body_tokens[0] != idx_tok:
        return [], False

    segments: list[list[int]] = []
    for tok in body_tokens:
        if tok == idx_tok:
            segments.append([])
        else:
            segments[-1].append(tok)

    parsed: list[AddressedTraceStep] = []
    for seg in segments:
        if len(seg) < 1 + body_len:
            return parsed, False
        addr = tuple(seg[:-body_len])
        body = seg[-body_len:]
        if use_query_tokens:
            pred_q0, pred_t0, pred_q1, pred_t1 = body
            if pred_q0 != int(q_0) or pred_q1 != int(q_1):
                return parsed, False
        else:
            pred_t0, pred_t1 = body
        sum_digit, carry_out = unpack_step_order(pred_t0, pred_t1, order)
        parsed.append(
            AddressedTraceStep(address=addr, sum_digit=sum_digit, carry_out=carry_out)
        )

    return parsed, halted_cleanly


@torch.no_grad()
def rollout_addressed_ar(
    model: Transformer,
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> list[Token]:
    """Run free-running greedy `addressed_ar` generation on `(alphabet, a, b)` until `[EOS]` or length cap.

    Starts with an empty trace after the third `[SEP]`, appends greedy `argmax`
    tokens from `SubspaceLMHead` up to the maximum well-formed target length for
    `max(len(a), len(b)) + 1` addressed columns plus `[EOS]`
    (`spec/variants/addressed_ar.md` §3), and stops as soon as `SpecialToken.EOS`
    is emitted (including `EOS` as the final element of the returned list when emitted).
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(
        alphabet, a, b, include_alphabet=include_alphabet
    )
    order = StepOrder(step_order)
    body_len = 4 if use_query_tokens else 2
    max_columns = max(len(a_tokens), len(b_tokens)) + 1
    max_tokens = (
        sum(
            1 + len(encode_address(k, alphabet_tokens)) + body_len
            for k in range(max_columns)
        )
        + 1
    )

    model.eval()
    device = model.device
    trace: list[Token] = []
    emitted: list[Token] = []

    for _ in range(max_tokens):
        example = build_addressed_ar_example(
            alphabet_tokens,
            a_tokens,
            b_tokens,
            trace,
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
                "rollout_addressed_ar requires a Transformer with SubspaceLMHead"
            )
        next_tok = int(logits[0, -1, :].argmax(dim=-1).item())
        emitted.append(next_tok)
        if next_tok == SpecialToken.EOS:
            break
        trace.append(next_tok)

    return emitted
